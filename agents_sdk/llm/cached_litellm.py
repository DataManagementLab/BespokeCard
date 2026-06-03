import asyncio
import copy
import logging
import time
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

import litellm
from agents import ModelSettings
from agents.extensions.models.litellm_model import LitellmModel
from litellm.exceptions import BadGatewayError, InternalServerError, RateLimitError

from agents_sdk.llm.cached_llm_helper import (
    LLMModelHelper,
)

logger = logging.getLogger(__name__)

_ANTHROPIC_CACHE_CONTROL_INJECTION_POINTS = [
    {"location": "message", "role": "system"},
    {"location": "message", "index": -1},
]


class LiteLLMCacheType:
    def __init__(
        self,
        response,
        parent_hash: str | None = None,
        hash_payload: str | None = None,
        llm_time: float | None = None,
    ):
        self.response = response
        self.parent_hash = parent_hash
        self.hash_payload = hash_payload
        self.llm_time = llm_time


class CachedLitellmModel(LitellmModel):
    def __init__(
        self,
        *args,
        llm_cache_dir: Path,
        do_not_cache: bool,
        working_dir: Path,
        stop_on_cache_miss: bool = False,
        tools_loaded_deferred: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cache_dir = llm_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stop_on_cache_miss = stop_on_cache_miss
        self.total_saved = 0.0
        self.llm_was_cached = False
        self.llm_model_helper = LLMModelHelper(
            model=self.model,
            cache_type=LiteLLMCacheType,
            do_not_cache=do_not_cache,
            is_litellm=True,
            tools_loaded_deferred=tools_loaded_deferred,
            working_dir=working_dir,
        )

    def _cache_path_for(self, hash: str) -> Path:
        return self.cache_dir / f"{hash}.pkl"

    def __str__(self) -> str:
        return str(self.model)

    def _is_anthropic_model(self) -> bool:
        return str(self.model).startswith("anthropic/")

    def _augment_model_settings_for_anthropic_prompt_caching(
        self, model_settings: ModelSettings
    ) -> Any:
        if not self._is_anthropic_model():
            return model_settings

        if model_settings is None:
            return model_settings

        # extra_body was deprecated by anthropic on 4/18/2026. Switching to writing to extra-args field

        try:
            extra_args = getattr(model_settings, "extra_args", None) or {}
            injection_points = extra_args.get("cache_control_injection_points")
            if injection_points:
                return model_settings

            updated_extra_args = dict(extra_args)
            updated_extra_args["cache_control_injection_points"] = (
                _ANTHROPIC_CACHE_CONTROL_INJECTION_POINTS
            )

            if is_dataclass(model_settings):
                return replace(model_settings, extra_args=updated_extra_args)  # type: ignore

            copied = copy.deepcopy(model_settings)
            setattr(copied, "extra_args", updated_extra_args)
            return copied
        except Exception as exc:
            logger.warning(
                "Failed to enable Anthropic prompt caching hints on model settings: %s",
                exc,
            )
            return model_settings

    async def get_response(self, *args, **kwargs):
        system_instructions = kwargs.get("system_instructions")
        input = kwargs.get("input")
        model_settings = kwargs.get("model_settings")
        tools = kwargs.get("tools") or []
        output_schema = kwargs.get("output_schema")
        handoffs = kwargs.get("handoffs") or []
        previous_response_id = kwargs.get("previous_response_id")
        conversation_id = kwargs.get("conversation_id")
        prompt = kwargs.get("prompt")

        assert model_settings is not None, "model_settings is required for caching"
        # model_settings.reasoning = {"effort": "low", "summary": "auto"}

        req_hash, hash_payload = self.llm_model_helper.hash_payload(
            system_instructions,
            input,
            model_settings,  # type: ignore
            tools=tools,
            output_schema=output_schema,
            handoffs=handoffs,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt,
        )

        cache_path = self._cache_path_for(req_hash)

        if cache_path.exists():
            resp, saved_cost, self.llm_was_cached = (
                self.llm_model_helper.load_llm_entry_from_cache(cache_path)
            )
            if resp is not None:
                self.total_saved += saved_cost
                return resp
        # get input tokens
        if self.stop_on_cache_miss:
            logger.debug(hash_payload)
            raise Exception(
                "Stop on cache miss. Did not found in cache: " + str(cache_path)
            )

        # add cache control injection points for Anthropic models to enable prompt caching
        kwargs["model_settings"] = (
            self._augment_model_settings_for_anthropic_prompt_caching(model_settings)
        )

        try:
            time_start = time.perf_counter()
            resp = await super().get_response(*args, **kwargs)
            llm_time = time.perf_counter() - time_start
        except RateLimitError as e:
            # wait (rate limit cooldown - at least one min)
            wait_min = 1  # minutes
            logger.warning(
                f"Rate limit error encountered: {e}. Waiting for {wait_min} minutes before retrying."
            )
            await asyncio.sleep(120)

            # try again
            time_start = time.perf_counter()
            resp = await super().get_response(*args, **kwargs)
            llm_time = time.perf_counter() - time_start
        except BadGatewayError as e:
            # wait
            wait_min = 1  # minutes
            logger.warning(
                f"Bad gateway error encountered: {e}. Waiting for {wait_min} minutes before retrying."
            )
            await asyncio.sleep(120)

            # try again
            time_start = time.perf_counter()
            resp = await super().get_response(*args, **kwargs)
            llm_time = time.perf_counter() - time_start
        except InternalServerError as e:
            if "overloaded_error" in str(e).lower():
                wait_min = 1  # minutes
                logger.warning(
                    f"Model server overloaded error encountered: {e}. Waiting for {wait_min} minutes before retrying."
                )
                await asyncio.sleep(120)

                # try again
                time_start = time.perf_counter()
                resp = await super().get_response(*args, **kwargs)
                llm_time = time.perf_counter() - time_start
            else:
                raise e
        except litellm.exceptions.Timeout as e:
            logger.warning(
                f"Timeout error encountered: {e}. This may be due to a transient issue with the model server. Retrying once after a short delay."
            )
            await asyncio.sleep(30)

            # try again
            time_start = time.perf_counter()
            resp = await super().get_response(*args, **kwargs)
            llm_time = time.perf_counter() - time_start

        # process response and cache it
        self.llm_model_helper.process_llm_response(
            resp=resp,
            llm_time=llm_time,
            cache_path=cache_path,
            hash_payload=hash_payload,
        )

        self.llm_was_cached = False

        return resp
