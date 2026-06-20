---
name: mysql-reset-password-interactive
overview: 以交互方式重置 MySQL 用户名和密码，用户自行输入新凭据
todos:
  - id: prompt-credentials
    content: 通过 Read-Host 交互式获取新用户名和密码
    status: completed
  - id: create-user-db
    content: 连接 MySQL 创建用户、数据库并授权
    status: completed
    dependencies:
      - prompt-credentials
  - id: verify-connect
    content: 用新凭据验证 MySQL 连接是否成功
    status: completed
    dependencies:
      - create-user-db
  - id: update-env
    content: 将新 DATABASE_URL 写入项目 .env 文件
    status: completed
    dependencies:
      - verify-connect
---

## 产品概述

用户希望重置 MySQL 的用户名和密码，但**不在计划阶段提供具体值**，而是在执行过程中通过**交互式命令行输入**新用户名和新密码。

## 核心功能

- 使用 PowerShell `Read-Host` 交互式提示用户输入**新的 MySQL 用户名**
- 使用 PowerShell `Read-Host -AsSecureString` 交互式提示用户输入**新的 MySQL 密码**（输入时字符被隐藏）
- 用当前可用的 root/root 凭据连接 MySQL 8.0 (localhost:3306)
- 创建新用户（或修改现有用户）并设置密码
- 创建数据库 `doc_converter`（如不存在）
- 授予新用户对该数据库的全部权限
- 验证新凭据可以成功连接
- 将生成的连接字符串写入项目 `.env` 文件的 `DATABASE_URL` 字段

## 技术栈

- **数据库**: MySQL 8.0 (已安装, 服务名 MySQL80, 路径 D:\ProgramData\MySQL\MySQL Server 8.0\bin\)
- **脚本环境**: PowerShell (Windows)
- **MySQL CLI**: mysql 命令行工具

## 实现方案

采用 **单步 PowerShell 脚本** 方案，通过管道将 SQL 命令传递给 mysql 客户端执行：

1. **交互式收集凭据**: 使用 `$newUsername = Read-Host "请输入新的 MySQL 用户名"` 和 `$newPassword = Read-Host "请输入新的 MySQL 密码" -AsSecureString` 获取用户输入
2. **创建用户 + 数据库 + 授权**: 通过 `mysql -u root -proot` 连接，执行以下 SQL：

- `CREATE USER IF NOT EXISTS 'xxx'@'localhost' IDENTIFIED BY 'yyy'`
- `CREATE DATABASE IF NOT EXISTS doc_converter`
- `GRANT ALL PRIVILEGES ON doc_converter.* TO 'xxx'@'localhost'`
- `FLUSH PRIVILEGES`

3. **验证连接**: 用新凭据执行测试查询 `SELECT 1`
4. **更新 .env**: 写入 `DATABASE_URL=mysql://用户名:密码@localhost:3306/doc_converter`

## 关键注意事项

- 密码使用 SecureString 收集，但在传递给 mysql 时需要转为明文（mysql CLI 不支持 SecureString 输入）
- 如果用户选择保留 root 用户名，则用 ALTER USER 代替 CREATE USER
- .env 文件位于 `c:\Users\Lenovo\Desktop\doc-converter\.env`

## 涉及文件

| 文件 | 操作 | 说明 |
| --- | --- | --- |
| `.env` | [MODIFY] | 更新 DATABASE_URL 字段为新的连接字符串 |