import zmq
import json
import logging
import threading
import time
import sqlite3
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class ZmqPublisher:
    """
    基于 ZeroMQ + SQLite 的持久化发布者
    无需 Redis，支持远程连接，支持消息回溯
    """
    def __init__(self, host=None, port=None, db_path=None):
        self.host = host or os.environ.get("ZMQ_HOST", "0.0.0.0")
        self.port = int(port or os.environ.get("ZMQ_PORT", 5555))
        self.db_path = db_path or os.environ.get("ZMQ_DB_PATH", "output/messages.db")
        
        # 确保数据库目录存在
        Path(os.path.dirname(self.db_path)).mkdir(parents=True, exist_ok=True)
        self._init_db()
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        # 设置 LINGER 为 0，确保进程退出时立即释放端口
        self.socket.setsockopt(zmq.LINGER, 0)
        try:
            self.socket.bind(f"tcp://{self.host}:{self.port}")
            logger.info(f"ZMQ Publisher bound to tcp://{self.host}:{self.port}")
        except zmq.ZMQError as e:
            if e.errno == zmq.EADDRINUSE:
                logger.warning(f"ZMQ Port {self.port} already in use. This is normal during Uvicorn hot-reload.")
            else:
                logger.error(f"Failed to bind ZMQ Publisher: {e}")
        except Exception as e:
            logger.error(f"Unexpected error binding ZMQ: {e}")

    def _init_db(self):
        """初始化 SQLite 数据库并优化性能"""
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            # 开启 WAL 模式提高并发性能
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT,
                    content TEXT,
                    timestamp REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_topic ON messages(topic)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON messages(timestamp)")
            
            # 开启自动清理机制的触发器：每插入 100 条消息，清理一次 24 小时前的消息
            # 或者更简单的：在 publish 时手动清理
            
    def _cleanup_old_messages(self, hours: int = 24):
        """清理旧消息，防止数据库无限增长"""
        try:
            threshold = time.time() - (hours * 3600)
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                conn.execute("DELETE FROM messages WHERE timestamp < ?", (threshold,))
                # 只有在大规模删除后才执行 VACUUM 以回收空间
                # 但 VACUUM 是阻塞的，所以这里只删除数据，SQLite 会重用这些空间
        except Exception as e:
            logger.error(f"Failed to cleanup old messages: {e}")

    def publish(self, topic: str, data: dict, persist: bool = True):
        """
        发布消息并持久化
        """
        # 每发布 50 条消息触发一次清理（24小时前的数据）
        if not hasattr(self, "_publish_count"):
            self._publish_count = 0
        self._publish_count += 1
        if self._publish_count % 50 == 0:
            threading.Thread(target=self._cleanup_old_messages, args=(24,), daemon=True).start()

        message_dict = {
            "topic": topic,
            "data": data,
            "timestamp": time.time()
        }
        content_json = json.dumps(message_dict, ensure_ascii=False)
        
        # 1. 持久化到 SQLite
        if persist:
            try:
                # 增加 timeout 处理高频写入冲突
                with sqlite3.connect(self.db_path, timeout=30) as conn:
                    conn.execute(
                        "INSERT INTO messages (topic, content, timestamp) VALUES (?, ?, ?)",
                        (topic, content_json, message_dict["timestamp"])
                    )
            except Exception as e:
                logger.error(f"Failed to persist message to SQLite: {e}")

        # 2. 通过 ZMQ 发布实时消息
        try:
            self.socket.send_string(f"{topic} {content_json}")
        except Exception as e:
            logger.error(f"Error publishing via ZMQ: {e}")

    def get_history(self, topic: str, since_ts: float = 0, limit: int = 100, delete_after: bool = False):
        """
        获取历史消息（供 API 调用）
        :param delete_after: 获取后是否从数据库删除这些消息
        """
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                # 1. 查询
                # 增加对 topic 为空的兼容性（如果 topic 是 ""，则查询所有）
                if topic:
                    query = "SELECT id, content FROM messages WHERE topic = ? AND timestamp > ? ORDER BY timestamp ASC LIMIT ?"
                    params = (topic, since_ts, limit)
                else:
                    query = "SELECT id, content FROM messages WHERE timestamp > ? ORDER BY timestamp ASC LIMIT ?"
                    params = (since_ts, limit)
                    
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                if not rows:
                    return []
                
                msg_ids = [row[0] for row in rows]
                messages = [json.loads(row[1]) for row in rows]
                
                # 2. 删除（如果需要）
                if delete_after:
                    # 使用批量删除优化
                    conn.execute(
                        f"DELETE FROM messages WHERE id IN ({','.join(['?']*len(msg_ids))})",
                        msg_ids
                    )
                
                return messages
        except Exception as e:
            logger.error(f"Failed to query message history: {e}")
            return []

class ZmqSubscriber:
    """
    基于 ZeroMQ 的状态订阅者（供前端或 EXE 使用）
    """
    def __init__(self, host=None, port=None, topics=None):
        self.host = host or os.environ.get("ZMQ_HOST", "127.0.0.1")
        self.port = int(port or os.environ.get("ZMQ_PORT", 5555))
        self.topics = topics or [""] # 默认订阅所有
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{self.host}:{self.port}")
        
        for topic in self.topics:
            self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        
        logger.info(f"ZMQ Subscriber connected to tcp://{self.host}:{self.port}")

    def receive(self, timeout=None):
        """
        接收消息
        """
        try:
            if timeout is not None:
                if not self.socket.poll(timeout):
                    return None
            
            message_str = self.socket.recv_string()
            # 拆分主题和内容
            topic, content = message_str.split(" ", 1)
            return json.loads(content)
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None

# 全局发布者单例
# 注意：在生产环境中，端口应可配置
messenger = ZmqPublisher()
