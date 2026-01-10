curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "你好，请介绍一下你自己", "thread_id": "test_user_001"}' \
     -N 