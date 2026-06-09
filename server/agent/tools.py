TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "todo_write",
            "description": "创建或更新任务列表，在动手前列出所有步骤，执行时逐步更新状态",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "完整的任务列表（每次调用必须包含全部任务）",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "任务描述"},
                                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"], "description": "任务状态"},
                            },
                            "required": ["content", "status"],
                        },
                    },
                },
                "required": ["todos"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "从矢量数据库中按时间区间查询采集的监控帧，返回帧列表（含时间戳和缩略图路径）",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "description": "开始时间，格式 YYYY-MM-DD HH:mm:ss"},
                    "end_time": {"type": "string", "description": "结束时间，格式 YYYY-MM-DD HH:mm:ss"},
                    "keywords": {"type": "array", "items": {"type": "string"}, "description": "搜索关键词，如吸烟、打架等"},
                },
                "required": ["start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sort_by_relevance",
            "description": "按匹配度对查询结果进行排序",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "待排序的原始数据"},
                    "query": {"type": "string", "description": "排序依据的查询描述"},
                },
                "required": ["data", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize",
            "description": "对最终结果进行总结表达",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "待总结的内容"},
                    "style": {"type": "string", "description": "表达风格，可选：简洁、详细、报告"},
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": "将重要信息保存到长期记忆。当用户说「记住」「请记住」或你想保存有价值的信息时调用",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "要记住的具体内容，应简洁明确"},
                    "type": {
                        "type": "string",
                        "enum": ["user", "feedback", "project", "reference"],
                        "description": "记忆类型：user=用户偏好, feedback=用户反馈, project=项目信息, reference=参考知识",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "关键词标签，用于后续检索匹配",
                    },
                },
                "required": ["content", "type"],
            },
        },
    },
]

TOOL_DESCRIPTIONS: dict[str, str] = {
    t["function"]["name"]: t["function"]["description"]
    for t in TOOLS
}

MAX_TOOL_ITERATIONS = 10
