import uuid
import os
from typing import Literal
from datetime import datetime
from trustcall import create_extractor
from typing import Optional 
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
from typing import TypedDict, Literal
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import merge_message_runs, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langchain_google_genai import ChatGoogleGenerativeAI
import os


# Load from google secrets
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
GEMINI = os.getenv("GOOGLE_API_KEY")

# Set variables for LangSmith Tracking
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langchain-academy"
os.environ["GEMINI_API_KEY"] = GEMINI

# User profile schema (Memory)
class Profile(BaseModel):
    """This is the profile of the user you are chatting with"""
    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="The user's location", default=None)
    target_audience: Optional[str] = Field(description="Target audience for content creation.", default=None)
    preferred_platforms: list[str] = Field(description="User's preferred platforms for posting content.", default_factory=list)

# Content schema (MemoryCollection)
class ContentCalendar(BaseModel):
    """Represents an item in the content calendar"""
    title: str = Field(description="The title or topic of the content.")
    platform: Optional[str] = Field(description="The platform where the content will be posted (e.g., Instagram, Blog).", default=None)
    deadline: Optional[datetime] = Field(description="The deadline for posting the content.", default=None)
    status: Literal["idea", "draft", "review", "posted"] = Field(description="The current status of the content.", default="idea")
    tags: list[str] = Field(description="Tags or topics associated with the content.", default_factory=list)
    idea: str = Field(description="General idea of the content item", default=None)

# Guidelines Prompt
CREATE_GUIDELINES = """Reflect on the following interaction.

Based on this interaction, update your guidelines for content creation.

Use any feedback from the user to update how they like to brainstorm, organize, or track content.

Your current instructions are:

<guidelines>
{guidelines}
</guidelines>"""




# Inspect the tool calls made by Trustcall
class Spy:
    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        # Collect information about the tool calls made by the extractor.
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                self.called_tools.append(
                    r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                )

# Initialize the spy
spy = Spy()

# Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0, api_key=GEMINI)

# Create the profile extractor
profile_extractor = create_extractor(
    model,
    tools=[Profile],
    tool_choice="Profile",
)

# Create the content extractor
content_extractor = create_extractor(
    model,
    tools=[ContentCalendar],
    tool_choice="ContentCalendar",
    enable_inserts=True).with_listeners(on_end=spy)


# Instruction
instruction = """Extract information about the content calendar using this conversation:"""

# Conversation
conversation = [HumanMessage(content="Hi, I'm Lore."),
                AIMessage(content="Nice to meet you, Lore."),
                HumanMessage(content="I want to make a Youtube video about LangGraph AI Agents.")]

# Invoke the extractor
result = content_extractor.invoke({"messages": [SystemMessage(content=instruction)] + conversation})

# Update the conversation
updated_conversation = [AIMessage(content="Oke great I added it to the content calendar"),
                        HumanMessage(content="Thanks! I want to publish it on Friday next week")]

# Update the instruction
system_msg = """Update existing memories and create new ones based on the following conversation:"""

# We'll save existing memories, giving them an ID, key (tool name), and value
tool_name = "ContentCalendar"
existing_memories = [(str(i), tool_name, memory.model_dump()) for i, memory in enumerate(result["responses"])] if result["responses"] else None

# Invoke the extractor with our updated conversation and existing memories
result = content_extractor.invoke({"messages": updated_conversation,
                                                        "existing": existing_memories})

def extract_tool_info(tool_calls, schema_name="Memory"):
    """Extract information from tool calls for both patches and new memories.

    Args:
        tool_calls: List of tool calls from the model
        schema_name: Name of the schema tool (e.g., "Memory", "ToDo", "Profile")
    """

    # Initialize list of changes
    changes = []

    for call_group in tool_calls:
        for call in call_group:
            if call['name'] == 'PatchDoc':
                changes.append({
                    'type': 'update',
                    'doc_id': call['args']['json_doc_id'],
                    'planned_edits': call['args']['planned_edits'],
                    'value': call['args']['patches'][0]['value']
                })
            elif call['name'] == schema_name:
                changes.append({
                    'type': 'new',
                    'value': call['args']
                })

    # Format results as a single string
    result_parts = []
    for change in changes:
        if change['type'] == 'update':
            result_parts.append(
                f"Document {change['doc_id']} updated:\n"
                f"Plan: {change['planned_edits']}\n"
                f"Added content: {change['value']}"
            )
        else:
            result_parts.append(
                f"New {schema_name} created:\n"
                f"Content: {change['value']}"
            )

    return "\n\n".join(result_parts)


# Inspect spy.called_tools to see exactly what happened during the extraction
schema_name = "ContentCalendar"
changes = extract_tool_info(spy.called_tools, schema_name)
print(changes)


# UpdateMemory schema
class UpdateMemory(TypedDict):
    """ Decision on what memory type to update """
    update_type: Literal['user', 'content_calendar', 'guidelines']

#main function
def content_generAItor(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]

    # Retrieve memories
    profile_memories = store.search(("profile", user_id))
    user_profile = profile_memories[0].value if profile_memories else None

    calendar_memories = store.search(("content_calendar", user_id))
    content_calendar = "\n".join(f"{item.value}" for item in calendar_memories)

    guidelines_memories = store.search(("guidelines", user_id))
    guidelines = guidelines_memories[0].value if guidelines_memories else ""

    # Prepare the system message
    system_msg = MODEL_SYSTEM_MESSAGE.format(
        user_profile=user_profile,
        content_calendar=content_calendar,
        guidelines=guidelines
    )

    # Generate a response
    response = model.bind_tools([UpdateMemory]).invoke( # Remove parallel_tool_calls=False
        [SystemMessage(content=system_msg)] + state["messages"]
    )
    return {"messages": [response]}

# Chatbot instruction for the content creation assistant
MODEL_SYSTEM_MESSAGE = """You are a helpful content creation assistant.

Your role is to help the user brainstorm, organize, and track content ideas for various platforms.

You maintain three types of memory:
1. The user's profile (general preferences for content creation).
2. A content calendar with ideas, deadlines, statuses, and drafts.
3. Guidelines for creating content based on user feedback.

Here is the current User Profile:
<user_profile>
{user_profile}
</user_profile>

Here is the current Content Calendar:
<content_calendar>
{content_calendar}
</content_calendar>

Here are the current content creation guidelines:
<guidelines>
{guidelines}
</guidelines>

Here are your instructions for reasoning about the user's messages:

1. Reason carefully about the user's messages as presented below.

2. Decide whether any of the your long-term memory should be updated:
- If personal information was provided about the user, update the user's profile by calling UpdateMemory tool with type `user`
- If content creations are mentioned, update the ContentCalendar list by calling UpdateMemory tool with type `content_calendar`
- If the user has specified preferences for how to update the ContentCalendar list, update the instructions by calling UpdateMemory tool with type `guidelines`

3. Tell the user that you have updated your memory, if appropriate:
- Do not tell the user you have updated the user's profile
- Tell the user them when you update the ContentCalendar list
- Do not tell the user that you have updated guidelines

4. Err on the side of updating the ContentCalendar list. No need to ask for explicit permission.

5. Respond naturally to user user after a tool call was made to save memories, or if no tool call was made.
"""

def route_message(state: MessagesState, config: RunnableConfig, store: BaseStore) -> Literal[END, "update_content_calendar", "update_guidelines", "update_profile"]:
    """Reflect on the memories and chat history to decide whether to update the memory collection."""
    message = state['messages'][-1]
    if len(message.tool_calls) == 0:
        return END
    else:
        tool_call = message.tool_calls[0]
        if tool_call['args']['update_type'] == "user":
            return "update_profile"
        elif tool_call['args']['update_type'] == "content_calendar":
            return "update_content_calendar"
        elif tool_call['args']['update_type'] == "guidelines":
            return "update_guidelines"
        else:
            raise ValueError

# Trustcall prompt
TRUSTCALL_INSTRUCTION = """Reflect on the following interaction.

Use the provided tools to retain any necessary memories about the user.

System Time: {time}"""

def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and update the memory collection."""

    # Get the user ID and define the namespace
    user_id = config["configurable"]["user_id"]
    namespace = ("profile", user_id)

    # Retrieve existing profile memories
    existing_items = store.search(namespace)
    tool_name = "Profile"
    existing_memories = ([(item.key, tool_name, item.value) for item in existing_items] if existing_items else None)

    # Format the instruction and merge with chat history
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    # Extract updates and save to the store
    result = profile_extractor.invoke({"messages": updated_messages, "existing": existing_memories})
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace, rmeta.get("json_doc_id", str(uuid.uuid4())), r.model_dump(mode="json"))

    tool_calls = state['messages'][-1].tool_calls
    return {"messages": [{"role": "tool", "content": "updated profile", "tool_call_id": tool_calls[0]['id']}]}

def update_content_calendar(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("content_calendar", user_id)

    existing_items = store.search(namespace)
    tool_name = "ContentCalendar"
    existing_memories = [(item.key, tool_name, item.value) for item in existing_items] if existing_items else None

    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = merge_message_runs([SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1])

    # Extract updates and save to the store
    result = content_extractor.invoke({"messages": updated_messages, "existing": existing_memories})

    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace, rmeta.get("json_doc_id", str(uuid.uuid4())), r.model_dump(mode="json"))
    tool_calls = state['messages'][-1].tool_calls

    # Extract the changes made by Trustcall and add the the ToolMessage returned to task_mAIstro
    content_calendar_update_msg = extract_tool_info(spy.called_tools, tool_name)
    return {"messages": [{"role": "tool", "content": content_calendar_update_msg, "tool_call_id":tool_calls[0]['id']}]}

# Instructions for updating the Content Calendar list
CREATE_INSTRUCTIONS = """Reflect on the following interaction.

Based on this interaction, update your instructions for how to update ContentCalendar list items.

Use any feedback from the user to update how they like to have items added, etc.

Your current guidelines are:

<guidelines>
{guidelines}
</guidelines>"""

def update_guidelines(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("guidelines", user_id)

    existing_memory = store.get(namespace, "content_guidelines")
    system_msg = CREATE_INSTRUCTIONS.format(guidelines=existing_memory.value if existing_memory else None)

    new_memory = model.invoke([SystemMessage(content=system_msg)] + state["messages"][:-1] + [HumanMessage(content="Please update the guidelines.")])
    store.put(namespace, "content_guidelines", {"memory": new_memory.content})

    tool_calls = state['messages'][-1].tool_calls
    return {"messages": [{"role": "tool", "content": "updated guidelines", "tool_call_id":tool_calls[0]['id']}]}

# Graph setup
builder = StateGraph(MessagesState)

# Add nodes to the graph
builder.add_node(content_generAItor)  # Central node for generating responses
builder.add_node(update_profile)  # Node for updating the user profile
builder.add_node(update_content_calendar)  # Node for updating the content calendar
builder.add_node(update_guidelines)  # Node for refining content guidelines

# Define the flow between nodes
builder.add_edge(START, "content_generAItor")  # Start at the content generator
builder.add_conditional_edges("content_generAItor", route_message)  # Route based on user input
builder.add_edge("update_profile", "content_generAItor")  # Return to generator after profile update
builder.add_edge("update_content_calendar", "content_generAItor")  # Return after calendar update
builder.add_edge("update_guidelines", "content_generAItor")  # Return after guidelines update

# Memory stores
across_thread_memory = InMemoryStore()
within_thread_memory = MemorySaver()

# Compile the graph
graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)