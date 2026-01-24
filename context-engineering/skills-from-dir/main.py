

import os
import yaml
import importlib.util
from pathlib import Path
from typing import Dict, Any, Callable

class Skill:
    def __init__(self, name: str, func: Callable, description: str, input_schema: dict):
        self.name = name
        self.func = func
        self.description = description
        self.input_schema = input_schema

class AgentSkillLoader:
    def __init__(self, skills_dir: str = "skills"):
        self.skills_dir = Path(skills_dir)
        self.registry: Dict[str, Skill] = {}

    def load_all_skills(self):
        """从 skills/ 目录加载所有技能子目录"""
        if not self.skills_dir.exists():
            raise FileNotFoundError(f"Skills directory not found: {self.skills_dir}")

        for skill_dir in self.skills_dir.iterdir():
            if skill_dir.is_dir():
                self._load_skill_from_dir(skill_dir)

    def _load_skill_from_dir(self, skill_dir: Path):
        """加载单个技能目录"""
        yaml_path = skill_dir / "skill.yaml"
        py_path = skill_dir / "skill.py"

        if not yaml_path.exists() or not py_path.exists():
            print(f"[Skip] Missing skill.yaml or skill.py in {skill_dir}")
            return

        # 1. 加载 YAML 元数据
        with open(yaml_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)

        # 2. 动态导入 Python 模块
        spec = importlib.util.spec_from_file_location(f"{skill_dir.name}_module", py_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 假设函数名与技能名一致
        func_name = meta["name"]
        if not hasattr(module, func_name):
            raise AttributeError(f"Function '{func_name}' not found in {py_path}")

        func = getattr(module, func_name)

        # 3. 创建 Skill 对象并注册
        skill = Skill(
            name=meta["name"],
            func=func,
            description=meta["description"],
            input_schema=meta["input_schema"]
        )
        self.registry[skill.name] = skill
        print(f"[Loaded] Skill: {skill.name}")

    def get_mcp_tools(self) -> list[dict]:
        """返回符合 MCP 协议的工具列表"""
        tools = []
        for skill in self.registry.values():
            tools.append({
                "name": skill.name,
                "description": skill.description,
                "inputSchema": skill.input_schema
            })
        return tools

    def invoke(self, name: str, arguments: dict) -> Any:
        """调用技能"""
        if name not in self.registry:
            raise ValueError(f"Skill '{name}' not registered.")
        return self.registry[name].func(**arguments)


# ----------------------------
# 主程序
# ----------------------------
if __name__ == "__main__":
    # 初始化并加载技能
    loader = AgentSkillLoader("skills")
    loader.load_all_skills()

    # 打印 MCP 工具列表（可发送给 LLM）
    print("\n[MCP Tools for LLM]:")
    import json
    print(json.dumps(loader.get_mcp_tools(), indent=2, ensure_ascii=False))

    # 模拟调用
    try:
        result = loader.invoke("get_weather", {"city": "Paris"})
        print(f"\n[Result]: {result}")
    except Exception as e:
        print(f"[Error]: {e}")