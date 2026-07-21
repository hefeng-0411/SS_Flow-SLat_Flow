# ① （强烈建议）将 pycache 加入忽略列表，避免误提交
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore

# ② 添加所有变更（包括新的 .gitignore）
git add .

# ③ 提交
git commit -m "chore: update .gitignore to exclude __pycache__"

# ④ 推送到远程 v2/main 分支
git push origin v2/main