# 部署指南 — 前端 (Vercel) 与 后端 (Render)

本文档说明如何使用 Vercel 部署前端 (client) 并使用 Render 部署后端（server）。

## 前端：Vercel（通过 GitHub 自动部署，推荐）

1. 登陆 https://vercel.com 并选择 "New Project" → "Import Git Repository"。
2. 选择你的仓库：`dekern7979/Public`。
3. 在 Project Settings：
   - Root Directory: `client`（如果你希望只部署客户端）
   - Framework Preset: `Vite`（或自动检测）
   - Build Command: 留空或使用 `pnpm build`（Vercel 将会安装依赖并执行）。
   - Output Directory: `dist` 或空（Vite 默认输出到 `dist`）。
4. 在 Environment Variables 中添加生产所需的变量（在 Vercel Project → Settings → Environment Variables）：
   - `VITE_FRONTEND_FORGE_API_KEY` = <your key>
   - `VITE_OAUTH_PORTAL_URL` = <url>
   - 其他前端需要的 `VITE_` 前缀变量
5. 点击 Deploy。部署完成后会生成一个公网 URL。

注意：如果你更希望部署根目录（同时构建前后端），可以把 Root Directory 设为 `/` 并使用 Build Command `pnpm build`，但这样部署会同时构建 server 的 bundle（通常我们把后端部署到专门的服务）。

## 后端：Render（推荐长期运行的 Node 服务）

1. 登录 https://render.com。
2. 在 Render 仪表板选择 "New" → "Web Service" → 连接 GitHub 仓库 `dekern7979/Public`。
3. 使用 `render.yaml`（本仓库已包含模板）或在向导中填写：
   - Build Command: `pnpm install && pnpm build`（或仅 `pnpm build`）
   - Start Command: `pnpm start`
   - Environment: Node
   - Health Check Path: `/api/health`（如 server 暴露）
4. 在 Render 的 Environment 区域添加机密（DATABASE_URL、AWS 凭证、JWT 秘钥等）。

## 本仓库变更

- 新增 `render.yaml`：Render 模板（请在 Render 控制台根据实际需填密钥）。
- 本文档 `DEPLOY.md`：部署步骤与注意事项。

## 完成后检查

- Vercel 上的前端 URL（确认页面加载且 API 调用指向正确后端域名）。
- Render 上的后端 URL（检查 `/api/health`）。

如需我代为完成 Vercel 仪表板的连接配置，我可以继续提供逐步截图和要在浏览器中点击的具体选项说明。

