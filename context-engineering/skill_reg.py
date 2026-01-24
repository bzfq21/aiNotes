import json
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, asdict

# ----------------------------
# 1. 定义 Skill 原始结构（用户编写的技能）
# ----------------------------

@dataclass
class RawSkill:
    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any]  # 简化的参数 schema

# 示例技能
def send_email(to: str, subject: str) -> str:
    return f"Email sent to {to} with subject: {subject}"

def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny, 25°C"

# 用户定义的原始技能包
RAW_SKILLS = [
    RawSkill(
        name="send_email",
        description="Send an email to a recipient",
        func=send_email,
        parameters={"to": "string", "subject": "string"}
    ),
    RawSkill(
        name="get_weather",
        description="Get current weather for a city",
        func=get_weather,
        parameters={"city": "string"}
    )
]

# ----------------------------
# 2. MCP 工具规范（简化版）
# ----------------------------

@dataclass
class MCTool:
    name: str
    description: str
    input_schema: Dict[str, Any]  # 符合 JSON Schema 子集
    # 注意：MCP 通常还包含 output_schema、context 等，此处简化

    def to_mcp_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": {
                    k: {"type": v} for k, v in self.input_schema.items()
                },
                "required": list(self.input_schema.keys())
            }
        }

# ----------------------------
# 3. Agent 的 Skill 加载器（解析 + 转换 + 注册）
# ----------------------------

class AgentSkillRegistry:
    def __init__(self):
        self._tool_registry: Dict[str, Callable] = {}  # 函数映射
        self._mcp_tools: List[MCTool] = []             # MCP 规范工具列表

    def load_skills_from_package(self, raw_skills: List[RawSkill]):
        """解析原始技能包，转换为 MCP 格式并注册到内存"""
        for skill in raw_skills:
            # 1. 转换为 MCP Tool
            mcp_tool = MCTool(
                name=skill.name,
                description=skill.description,
                input_schema=skill.parameters
            )
            self._mcp_tools.append(mcp_tool)

            # 2. 注册可调用函数（字典映射）
            self._tool_registry[skill.name] = skill.func

        print(f"[INFO] Loaded {len(raw_skills)} skills into registry.")

    def get_mcp_tool_list(self) -> List[Dict[str, Any]]:
        """返回符合 MCP 协议的工具列表（供 LLM 使用）"""
        return [tool.to_mcp_dict() for tool in self._mcp_tools]

    def invoke_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """执行工具调用"""
        if tool_name not in self._tool_registry:
            raise ValueError(f"Tool '{tool_name}' not found.")
        func = self._tool_registry[tool_name]
        return func(**arguments)

# ----------------------------
# 4. 使用示例
# ----------------------------

if __name__ == "__main__":
    # 初始化 Agent 注册器
    agent = AgentSkillRegistry()

    # 加载技能包（模拟从文件/模块加载）
    agent.load_skills_from_package(RAW_SKILLS)

    # 获取 MCP 兼容的工具列表（可发送给 LLM）
    mcp_tools = agent.get_mcp_tool_list()
    print("\n[MCP Tools for LLM]:")
    print(json.dumps(mcp_tools, indent=2))

    # 模拟 LLM 决定调用 get_weather
    tool_call = {"name": "get_weather", "arguments": {"city": "Beijing"}}
    result = agent.invoke_tool(tool_call["name"], tool_call["arguments"])
    print(f"\n[Tool Result]: {result}")