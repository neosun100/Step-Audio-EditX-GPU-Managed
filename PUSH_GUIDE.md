# 推送指南

## 📦 待推送内容

当前有 **8 个提交** 等待推送到 https://github.com/neosun100/Step-Audio-EditX

## 📋 提交列表

```bash
9a334f9 docs: 添加更新总结文档
d664ea0 merge: 合并远程更新并添加统一部署功能
dce51bf feat: 统一部署架构 - UI和API共享模型实例
151453f feat: 添加模型懒加载和统一容器部署
e15d2be docs: 添加最佳实践章节
a468623 docs: 添加最佳实践章节
...
```

## 🚀 推送方法

### 方法1：直接推送（推荐）

```bash
cd /home/neo/upload/Step-Audio-EditX
git push neosun100 main
```

### 方法2：强制推送（如有冲突）

```bash
cd /home/neo/upload/Step-Audio-EditX
git push neosun100 main --force
```

### 方法3：使用 SSH（如 HTTPS 超时）

```bash
# 修改 remote URL 为 SSH
git remote set-url neosun100 git@github.com:neosun100/Step-Audio-EditX.git

# 推送
git push neosun100 main
```

### 方法4：分批推送（如文件太大）

```bash
# 推送最近3个提交
git push neosun100 HEAD~5:main

# 然后推送剩余的
git push neosun100 main
```

## 📝 主要更新内容

### 1. 统一部署架构
- `app.py` - 添加 API 支持
- `start_unified.sh` - 统一启动脚本
- `unified_app.py` - 备用入口

### 2. 文档更新
- `README.md` - 添加统一部署说明
- `UPDATE_SUMMARY.md` - 更新总结
- `COMPLETION_REPORT.md` - 完成报告
- `QUICK_START.md` - 快速开始

### 3. 优化功能
- `lazy_model_manager.py` - 懒加载管理器
- `server.py` - 优化服务器
- 修复 `start_ui_container.sh` 参数问题

## ⚠️ 注意事项

1. **模型文件**：已在 .gitignore 中排除大文件
2. **缓存文件**：cache/ 目录已排除
3. **文档完整**：所有文档已更新

## ✅ 验证推送

推送成功后，访问：
https://github.com/neosun100/Step-Audio-EditX

检查：
- ✅ README.md 已更新
- ✅ start_unified.sh 存在
- ✅ UPDATE_SUMMARY.md 存在
- ✅ 提交历史正确

## 🔧 故障排查

### 问题1：408 超时

**原因**：网络不稳定或文件太大

**解决**：
```bash
# 增加缓冲区
git config http.postBuffer 524288000

# 重试
git push neosun100 main
```

### 问题2：403 权限错误

**原因**：认证失败

**解决**：
```bash
# 检查凭据
git config --list | grep credential

# 重新设置
git config credential.helper store
```

### 问题3：冲突

**原因**：远程有新提交

**解决**：
```bash
# 拉取并合并
git pull neosun100 main --rebase

# 推送
git push neosun100 main
```

## 📞 获取帮助

如果推送失败，可以：
1. 查看详细错误：`git push neosun100 main --verbose`
2. 检查网络：`ping github.com`
3. 使用 GitHub Desktop 或其他 GUI 工具

---

**准备时间**：2025-12-04 01:54
**状态**：✅ 已准备就绪
