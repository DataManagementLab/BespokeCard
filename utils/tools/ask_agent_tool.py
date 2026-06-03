import logging
from typing import Dict, Any


logger = logging.getLogger(__name__)


class AskAgentTool:
    def __init__(self, target_agent):
        self.target_agent = target_agent
        pass

    async def __call__(self, message: str) -> str:
        """Sends a message to the target agent and returns their reply."""
        # We limit max_turns to prevent infinite loops between agents
        logger.info(f"Invoking AskAgentTool with message: {message[:100]}...")
        question = f"This is the coding agent, I have the following question:\n{message}.\nDo not make compromises regarding the proposed statistics without a good reason (e.g., computational complexity). In that case, propose a clear alternative. Answer compactly but with enough detail to be actionable for the coding agent."
        response = await self.target_agent.run_agent(
            prompt=question, max_turns=10, short_desc=""
        )
        logger.info(f"Received response from target agent: {response[:100]}...")
        return response
