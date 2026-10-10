# kb-index

[English](README.md) | 简体中文

把多个 Markdown 文档仓索引进一个 SQLite 文件，做向量 + BM25 混合检索，通过 MCP 提供给 AI 编程助手。

我的笔记、工作文档和几个项目 wiki 分散在不同的仓里，每个仓有自己的 `AGENTS.md` 规则。AI 助手总在问
"这事在哪个仓、该守哪份规则"，也总找不到和问题用词不同的文档。kb-index 把它们全部索引进一个 `index.db`，
给助手四个只读工具。每条结果都带着它所在仓的规则：规则跟着内容走，而不是取决于助手是在哪个目录里启动的。

```
 文档仓（git 仓，裸仓也行 / 普通目录）
        │  idx index     （增量；定时跑一次容器）
        ▼
   ┌──────────┐   SQLite：chunks + 向量 + FTS5 + 链接图 + meta（规则、客户端设置）
   │ index.db │   发布时附 sha256、manifest 和逐文件增量包
   └──────────┘
        │  用什么方式分发到各机都行（scp、共享目录、`ssh host cat` ……）
        ▼
   kb-index-mcp  （每台机一个进程，回环地址上的 HTTP；或 stdio）
        │
        ▼
   Claude Code / Codex / 任意 MCP 客户端
```

心智模型：md 文件是源码，`index.db` 是编译产物，索引器是编译器。写文档不需要任何额外动作；库坏了删掉重建即可。

## 能做什么

- **混合检索**：向量腿（余弦相似度）+ BM25 腿（SQLite FTS5，trigram 分词，中文可用），用 RRF 融合。
  相似度阈值只砍向量腿：`8081`、容器名这类标识符即使 embedding 完全没用，也能被 BM25 找到。两条腿都空时
  老实返回 `found: false`。
- **规则跟着内容走**：每个源有一句 `rule`；每个工具都返回命中仓的规则，外加 `agents_chain`——从仓根到命中
  文件逐层存在的 `AGENTS.md`，告诉助手动手前该读哪几份。
- **免费的链接图**：`[[wiki 链接]]` 和 md 相对链接解析成边。检索命中自带 `links`（被谁链 / 链出去谁），
  `archive/` 目录下的文档、文件名带 old / deprecated 的、小节标题是"已废弃 / deprecated"的，带
  `status: archived`（中英双语词表）。
- **知道自己不知道什么**：`.indexignore` 规则随库发布；检索没命中时作为 `excluded` 返回，助手分得清
  "库里没有"和"故意不进索引、去本地仓找"。
- **全程增量**：git 源按 blob sha 比对（没变的文件连读都不读），目录按 mtime + 大小 + 哈希；向量按文本哈希
  缓存，改切块或重建都不重复付费；发布时生成逐文件增量包，客户端通常只下载几 MB。
- **embedding 可插拔**：Gemini、任意 OpenAI 兼容接口（OpenAI、Ollama、LM Studio、vLLM、TEI、多数云厂商的兼容
  模式），或离线的 `hash`（没有 key 也能先跑通）。不依赖 SDK，不引入 torch。

## 快速开始

需要 Python 3.12+ 和 git。

```bash
git clone https://github.com/ZhengchenTao/kb-index.git && cd kb-index
pip install -e .

# 用离线 hash provider 索引自带的示例语料（不需要 API key）
idx --config examples/sources.example.yaml --db /tmp/kb/work.db index
idx --db /tmp/kb/index.db query "how do I roll back a release" -v
idx --db /tmp/kb/index.db grep 8081
idx --db /tmp/kb/index.db neighbors deploy.md
idx --db /tmp/kb/index.db grep quarterly-tax-reminder      # 被排除的文件：没有匹配，excluded 说明原因

# 起 MCP
kb-index-mcp --db /tmp/kb/index.db                          # stdio
kb-index-mcp --db /tmp/kb/index.db --http 127.0.0.1:47291   # 全机共用的 HTTP 服务
```

Windows 上 `Scripts` 目录常不在 PATH 里，`python -m kbindex.cli` / `python -m kbindex.mcp_server` 在哪都能用。

正式使用时把 `examples/sources.example.yaml` 复制到**本仓之外**（配置里有私有路径和规则，应该跟着你的部署走），
换成真正的 embedding provider，把 `sources` 指向你的文档仓。

### 接入 MCP 客户端

Claude Code（`~/.claude.json` 或 `claude mcp add`）：

```json
"kb-index": { "type": "http", "url": "http://127.0.0.1:47291/mcp" }
```

Codex（`~/.codex/config.toml`）：

```toml
[mcp_servers.kb-index]
url = "http://127.0.0.1:47291/mcp"
```

或 stdio：`{"command": "kb-index-mcp", "args": ["--db", "/path/to/index.db"]}`。

服务进程的环境里要有查询侧的 API key（变量名由 `embedding.api_key_env` 决定，默认 `GEMINI_API_KEY`），
别的都不需要：规则、阈值、排除规则、客户端设置都从库里读。

## 四个工具

| 工具 | 用途 | 返回 |
|---|---|---|
| `search(query, k, repos?, domains?, expand?)` | 自然语言提问 | top-k 定位（repo / path / heading / 片段 / 分数）、`links`、`status`、`repo_rules`、`repos`、`agents_chain`；没命中时带 `excluded` |
| `grep(pattern, k, repos?, domains?)` | 端口、容器名、路径、标识符（正则，不区分大小写） | 命中行及所在文档，同样带规则 / 仓信息 / AGENTS 链 |
| `neighbors(path, repo?)` | 一篇文档的完整链接关系 | 链出 / 链入（每条带 `status`）、语义相近的文档 |
| `read_doc(path, repo?)` | 读本机没有 clone 的仓里的文档 | 由 chunk 重组的正文（不是逐字原文）和该仓规则 |

工具只返回定位，不返回全文；工具说明会提示助手本地有仓时直接读原文件。

## 配置

索引器读 `sources.yaml`（`--config` > `$KB_CONFIG` > `./sources.yaml`）。客户端从不读它：它们需要的东西都写在
`index.db` 的 `meta` 表里。字段说明见英文 README 的「Configuration」一节（字段名相同，注释逐条对应）：

- `embedding`：`provider`（gemini / openai / hash）、`model`、`dim`、`batch`、`api_key_env`（存的是变量名，
  key 本身永不进库）、`base_url`、`query_prefix` / `document_prefix`（非对称模型的指令前缀）。
- `search`：`threshold`（向量腿的余弦下限，跟模型强相关，用 `idx calibrate` 标定）、`rrf_k`。
- `chunk`：`min`、`max`、`preamble_label`（第一个标题之前那段文字的标题名，默认 `(preamble)`）。
- `ignore_file`（相对配置文件）、`extended_meta`（见下文「线上升级」）。
- `client`：随库下发给各机 MCP 的设置——`workspace`（本机仓目录根，按 `nt` / `posix` 分别给，用于
  `local_path` / `local_exists` / `clone_hint`）、`agents_files`、`freshness_note`、`instructions`、`tool_descriptions`。
- `sources[]`：`repo`、`type`（git / dir）、`path`（绝对路径，或相对配置文件）、`ref`、`include`、`domain`、
  `domains`（路径前缀 → domain，最长前缀优先）、`rule`、`url`、`clone_hint`。

环境变量：`KB_DB`（工作库，默认 `./work.db`）、`KB_OUT`（发布目录，默认工作库所在目录）、`KB_INDEXIGNORE`、
`KB_WORKSPACE`（在某台机上覆盖 `client.workspace`）、`KB_USAGE_LOG`。

### `.indexignore`

gitignore 风格，匹配对象是 `repo/仓内相对路径`：`**` 跨目录，`*` 不跨 `/`，末尾 `/` 表示目录，不含 `/` 的裸模式匹配
任意层级的文件名。`idx ignore` 列出每条规则命中了哪些文件。**进了索引的内容，库的每个使用者都能读到**：密钥、凭据、
个人数据用这里的规则挡在外面，而且记住文件名本身也会泄露信息。

## 命令

```
idx index [--repo R] [--force] [--dry-run] [--no-publish] [--publish-always]
idx rebuild [--repo R] [--purge-cache]   # 清掉 chunk 重切；向量缓存默认复用（免费），--purge-cache 才重新付费
idx status | ignore | publish
idx query "..." [-k 5] [--repo R] [--domain D] [--expand] [-v]   # 打印两条腿的分数
idx grep PATTERN | neighbors PATH | read PATH
idx calibrate --queries q.yaml          # 标定阈值
idx sig --db FILE                        # 版本号 + 内容签名（拉库脚本用）
idx apply-delta --db FILE DELTA.db.gz ...
```

`idx index` 只在客户端读得到的东西真的变了（chunk、向量、规则、URL、阈值、排除规则、客户端设置）时才发布，
否则打印 `unchanged vs published index.db, not published`。每次发布都是一个新版本、所有客户端都要拉，空跑不能产生版本。

### 标定阈值

余弦阈值完全取决于模型（和维度）。写几条库里能答的问题、几条答不了的问题（要包括通用技术题，误命中主要来自它们），
`idx calibrate --queries q.yaml` 在关掉阈值的情况下打印每条的 top-1 余弦，并建议取"不在库最高"与"在库最低"的中点。
`gemini-embedding-001` 1536 维时，20 条真实查询在库 ≥ 0.727、不在库 ≤ 0.649，所以取 0.69。换模型后必须重测。

## 分发数据库

索引器往 `$KB_OUT` 写：`index.db`（剔掉了向量缓存）、`index.db.sha256`、`deltas/<N>.db.gz`（版本 N-1 → N 的
增量，保留最近 40 个）、`index.manifest.json`（格式见英文 README）。

怎么搬文件由你决定（例如 `ssh host cat`，服务器连不到客户端时也能用）。拉库脚本应当：

1. 读 `index.manifest.json`；本地 `idx sig --db local.db` 的签名等于 `sig` 就结束。
2. 否则从本地版本 + 1 走到 `version`，每个增量的 `from_sig` 必须等于走到这里的签名。链完整就下载这些包、逐个校验
   `sha256`，在副本上 `idx apply-delta --db copy.db <包...>`，打印出的签名等于 `sig` 才替换本地文件。
3. 链断了（本地太旧、分叉、哈希不对）：整库下载 `index.db` 并校验 `index.db.sha256`。

签名不是文件哈希（打过增量的文件和发布物字节不同），而是对每个文件在库里存的全部内容（源文件哈希、chunk id、domain、
标题、文本哈希、结构边、向量字节、链接边）做摘要，再加上全部 meta 行算 sha256。这个算法是发布端与客户端之间的协议：
改了它，所有客户端都会退回整库下载。

替换数据库要原子（先写临时文件再改名）。MCP 每次调用开一次连接，文件 mtime 变了就重载内存里的向量，不用重启。
Windows 上调用期间文件被占用时改名会失败，重试几秒即可。

## 运行

**索引器**：`Dockerfile` 构建入口为 `idx` 的镜像；配置挂到 `/config`，文档仓只读挂载，数据卷挂到 `/data`
（见 `examples/docker-compose.example.yml`），用 cron 或计划任务跑 `docker compose run --rm kb-index index`。
git 服务器的裸仓一律只读挂载，绝不在里面执行任何写操作：别的用户写进去的对象可能让 git 服务器自己再也写不进去。

**MCP 服务**：每台机一个 HTTP 进程（`--http`，只绑回环地址）给所有会话共用，比每个会话起一个 stdio 进程好：

- numpy 加载前设 `OPENBLAS_NUM_THREADS=1`。否则 OpenBLAS 给每个线程预提交一块缓冲区：28 核机器上每个进程约
  900 MB 提交量，十几个会话就把提交上限打满；单线程后约 175 MB，点积仍是毫秒级，远小于一次约 900 ms 的 embedding 请求。
- 无状态 streamable HTTP + JSON 响应：服务重启对已连接的客户端透明。
- 端口即单实例锁（Windows 上 `SO_EXCLUSIVEADDRUSE`），重复启动的实例以退出码 2 退出。
- 回环地址上自动开启 DNS rebinding 防护。本机任何进程都能调用它，而它暴露整个语料库：绝不要绑到对外网卡。
- `--log FILE` 记录每次失败和每次没命中，事后查得到"助手说 kb 没搜到"的原因（额度用尽、断网、没注入 key）。
- 用系统自带的方式守护它（systemd、launchd、会自动重启它的计划任务）。

**用量记账**：Gemini 的 embedding 响应不带 token 数，API key 也查不了用量，所以每个请求往 `embed-usage.jsonl`
追加一行（`$KB_USAGE_LOG`，否则 `$KB_DB` 同目录，否则不记）：
`{"ts", "date", "kind": "index"|"query", "model", "tokens", "items", "chars", "exact"}`。文件按月裁剪而不是按行数，
据此算出的当月累计不会凭空变小。

## 线上升级

| 改动 | 影响 |
|---|---|
| `rule`、`url`、`clone_hint`、`client.*`、`.indexignore`、`threshold` | 只改 meta：发布一个很小的增量 |
| `domain` / `domains` | 下次 `idx rebuild --repo R` 时重写 chunk，向量从缓存取（免费） |
| `chunk.min` / `max` / `preamble_label`、切块代码 | chunk 文本变了：重新 embed（花钱）、增量包很大 |
| `model`、`dim`、`provider`、`document_prefix` | 全部重新 embed：`idx rebuild --purge-cache`，然后重新标定 |

`extended_meta: false` 只写旧版本就有的那几类 meta（`model`、`dim`、`threshold`、`rrf_k`、`ignore`、`rule:*`、`url:*`），
用于升级正在运行的部署：先关着它部署新代码，确认索引器打印 `unchanged ..., not published`（证明客户端看到的东西一点没变），
再打开。

## 设计取舍

- **SQLite + numpy，不用向量数据库**：单文件、无常驻进程、拷贝即分发。16k × 1536 的 float32 暴力点积几毫秒；
  语料再大十倍再考虑 sqlite-vec 或 ANN 索引。
- **按标题切块**：文档本来就有结构，标题是天然的语义边界，命中标签也可读（`文件 > 一级标题 > 二级标题`）。
  超长小节先按段落、再按行硬切（表格和代码块里没有空行）；过短的块丢掉。
- **每轮分两段**：先把缺的向量全部 embed、逐批写进缓存并提交，再写 chunk。中途崩了重跑不重复付费。
- **关系注解塞进命中结果，而不是单独的工具**：实际工作中助手几乎不会去调 neighbors，拿到路径就直接读文件了；
  链接和归档信号只有跟着命中一起回来才会被用上。
- **非对称 embedding**：Gemini 对文档和查询用不同的 task type，query↔doc 与 doc↔doc 的余弦不是一个尺度，所以阈值必须用
  真实查询标定。

## 开发

```bash
pip install -e ".[test]"
python -m pytest
```

`tests/golden_examples.json` 冻结了切块器在示例语料上的输出。切块是与所有现有部署的契约（一改就要重新 embed 它们的语料）；
确实要改时用 `PYTHONPATH=. python tests/test_chunker.py --update` 重新生成，并在 commit message 里写明。

## 安全

- `index.db` 含有全部被索引内容的全文，敏感等级等于其中最敏感的那篇文档。
- MCP 服务只读，只应监听回环地址。
- API key 只从环境变量读；`index.db` 里存的是变量名，永远不是值。

## 许可证

MIT
