
# Get Weather Skill

## 功能
获取全球主要城市的当前模拟天气。

## 输入
- `city` (string): 城市名称，支持中英文。

## 输出
- 字符串格式的天气描述，例如：`"Weather in Beijing: Sunny, 25°C"`

## 注意
- 本技能为模拟实现，不调用真实 API。
- 不支持县级或街道级地名。

## 示例
```json
{ "city": "Tokyo" }
```