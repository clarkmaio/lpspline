# AI Agent Instructions

You have been tasked with working on this repository. This document outlines the standards, workflows, and philosophies you should follow to ensure high-quality contributions.

## Core Philosophy

1.  **Think Before You Code**: Always analyze the request and the existing codebase before making changes. Understand the *why* and *how*.
2.  **Be a Good Boy Scout**: Leave the code better than you found it. If you see a small, unrelated issue (like a typo or bad indentation) that is safe to fix, fix it.
3.  **Quality Over Speed**: It is better to write correct, maintainable, and well-tested code than to be fast but sloppy.

## Workflow

1.  **Context Loading**:
    *   Read `README.md` to understand the project's purpose.
    *   Read related files to understand the current implementation.
    *   Check for existing tests to ensure you don't break functionality.

2.  **Planning**:
    *   For complex tasks, create a short implementation plan. This helps you structure your thoughts and allows the user to review your approach.
    *   Identify potential risks or breaking changes.

3.  **Implementation**:
    *   Write clean, readable code.
    *   Follow the **Coding Standards** below.
    *   Keep changes focused on the task at hand.

4.  **Verification**:
    *   **Always runs tests**. If tests don't exist, create them.
    *   Verify your changes manually if possible (e.g., run the script, check the output).
    *   Ensure all tests pass before finishing the task.

## Coding Standards

### Python
*   **Style**: Follow [PEP 8](https://peps.python.org/pep-0008/).
*   **Docstrings**: Use [Google Style Python Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings). Every module, class, and public function should have a docstring.
*   **Type Hints**: Use type hints for function arguments and return values. This improves readability and tooling support.
*   **Imports**: Sort imports using `isort` conventions (standard library, third-party, local).
*   **Complexity**: Keep functions short and focused. If a function does too much, refactor it.

### Tests
*   We use `pytest` for testing.
*   Write unit tests for all new logic.
*   Mock external dependencies where appropriate to keep tests fast and reliable.

## Communication
*   Be clear and concise in your explanations.
*   Explain *why* you made certain decisions, not just *what* you did.
*   If you are unsure about something, ask the user for clarification.

## Security
*   Never commit or push secret variables such as passwords, token, api keys....
*   Ignore .env file. Never directly read it.
*   Access to it must be done through environment variables.
*   Never delete or modify files out of the package directory.
*   If user ask to delete or modify files out of the package directory, ask for confirmation.