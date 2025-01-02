from pydantic_ai import Agent, RunContext
import lib_models as model_lib

roulette_agent = Agent(
    # model_lib.model_ollama,
    model_lib.model_openai,
    deps_type=int,
    result_type=bool,
    system_prompt=(
        "Use the `roulette_wheel` function to see if the "
        "customer has won based on the number they provide."
    ),
)


@roulette_agent.tool
async def roulette_wheel(ctx: RunContext[int], square: int) -> str:
    """check if the square is a winner"""
    return "winner" if square == ctx.deps else "loser"


# Run the agent
success_number = 18
result = roulette_agent.run_sync("Put my money on square eighteen", deps=success_number)
print(result.data)
# > True

result = roulette_agent.run_sync("I bet five is the winner", deps=success_number)
print(result.data)
# > False


'''
Only for 'openai:gpt-4o'
if using llama 3.2, have exception error. pydantic_ai.exceptions.UnexpectedModelBehavior: Exceeded maximum retries (1) for result validation
'''
