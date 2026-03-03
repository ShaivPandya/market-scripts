# Command Briefing Skill

## Overview
This skill ensures the user is always informed about what the agent is doing in the terminal before any commands are executed.

## Instructions
Before using the `run_command` tool, you MUST:
1. **Explain the Command**: Provide a concise summary of what the specific command string does (e.g., "This checks disk usage for hidden files").
2. **State the Purpose**: Explain *why* you are running it in the current context (e.g., "I'm checking this to see if a large repo size is causing the GitHub 500 error").
3. **Wait for Approval**: Unless the command is explicitly marked as `SafeToAutoRun: true` and is a non-destructive read-only operation, clearly ask for the user's permission to proceed.

Even for "SafeToAutoRun" commands, a brief one-line explanation should precede the tool call in your response.
