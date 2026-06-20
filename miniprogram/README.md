# 文档转换器微信小程序（Web-View版）

## 🚀 快速上线步骤

### 第一步：下载并安装 cpolar 内网穿透
1. 访问 https://www.cpolar.com/download
2. 下载 Windows 版本并安装
3. 注册账号并获取认证 token

### 第二步：启动本地后端服务
确保您的后端服务正在运行（端口3001）：
```bash
# 在项目根目录
pnpm dev
```

### 第三步：用 cpolar 映射本地端口
打开 cpolar，运行：
```bash
cpolar http 3001
```
复制显示的 HTTPS 公网地址（例如 `https://xxx.cpolar.cn`）

### 第四步：配置小程序 URL
打开 `pages/index/index.js`，替换 `webUrl` 为 cpolar 的公网地址：
```javascript
data: {
  webUrl: "https://您的cpolar地址.cpolar.cn"
}
```

### 第五步：用微信开发者工具打开项目
1. 下载微信开发者工具：https://developers.weixin.qq.com/miniprogram/dev/devtools/download.html
2. 导入项目，选择 `miniprogram` 文件夹
3. 点击"编译"，就能看到文档转换器了！

## 📌 注意事项

- Web-View 需要配置业务域名（正式发布时）
- 开发测试阶段可以在微信开发者工具中关闭"不校验合法域名"
