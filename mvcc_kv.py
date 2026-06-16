import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Set, Tuple
from collections import OrderedDict
import uuid


@dataclass
class Version:
    """
    MVCC 版本节点
    
    每个 key 维护一个版本链表，每个版本记录：
    - value: 该版本存储的值
    - create_ts: 创建该版本的事务的提交时间戳
    - expire_ts: 该版本被覆盖的时间戳（None 表示这是当前最新版本）
    """
    value: Any
    create_ts: int
    expire_ts: Optional[int] = None


@dataclass
class Transaction:
    """
    事务上下文
    
    每个事务拥有：
    - txn_id: 全局唯一事务ID
    - read_ts: 快照时间戳，决定该事务能看到哪些版本
    - status: 事务状态 (active, committed, aborted)
    """
    txn_id: str
    read_ts: int
    status: str = "active"
    writes: Dict[str, Any] = field(default_factory=dict)


class TransactionManager:
    """
    事务管理器
    
    核心职责：
    1. 分配单调递增的时间戳（用于 read_ts 和 commit_ts）
    2. 追踪所有活跃事务及其 read_ts
    3. 计算低水位（Low Water Mark）- 所有活跃事务中最小的 read_ts
    4. 低水位是GC的依据：所有 expire_ts < low_water_mark 的版本都可安全回收
    """
    
    def __init__(self):
        self._timestamp = 0
        self._ts_lock = threading.Lock()
        self._active_txns: Dict[str, Transaction] = {}
        self._active_lock = threading.Lock()
        self._low_water_mark = 0
    
    def allocate_ts(self) -> int:
        """分配下一个单调递增的时间戳"""
        with self._ts_lock:
            self._timestamp += 1
            return self._timestamp
    
    def begin_txn(self) -> Transaction:
        """
        开始一个新事务
        
        1. 分配 read_ts（快照时间戳）
        2. 将事务加入活跃事务表
        3. 更新低水位
        """
        read_ts = self.allocate_ts()
        txn_id = str(uuid.uuid4())
        txn = Transaction(txn_id=txn_id, read_ts=read_ts)
        
        with self._active_lock:
            self._active_txns[txn_id] = txn
            self._update_low_water_mark()
        
        return txn
    
    def end_txn(self, txn: Transaction, status: str = "committed"):
        """
        结束一个事务
        
        1. 更新事务状态
        2. 从活跃事务表移除
        3. 更新低水位（可能上升）
        """
        txn.status = status
        with self._active_lock:
            if txn.txn_id in self._active_txns:
                del self._active_txns[txn.txn_id]
                self._update_low_water_mark()
    
    def _update_low_water_mark(self):
        """
        更新低水位（Low Water Mark）
        
        低水位定义：所有活跃事务中最小的 read_ts
        如果没有活跃事务，低水位 = 当前最新时间戳
        
        这是GC的关键：
        - 任何版本的 expire_ts < low_water_mark 都可以安全回收
        - 因为不存在任何活跃事务能看到这些版本
        """
        if not self._active_txns:
            with self._ts_lock:
                self._low_water_mark = self._timestamp
        else:
            min_read_ts = min(t.read_ts for t in self._active_txns.values())
            self._low_water_mark = min_read_ts
    
    def get_low_water_mark(self) -> int:
        """获取当前低水位（线程安全）"""
        with self._active_lock:
            return self._low_water_mark
    
    def get_active_txn_count(self) -> int:
        """获取活跃事务数量"""
        with self._active_lock:
            return len(self._active_txns)
    
    def get_active_read_timestamps(self) -> Set[int]:
        """获取所有活跃事务的 read_ts（用于调试）"""
        with self._active_lock:
            return set(t.read_ts for t in self._active_txns.values())


class MVCCKVStore:
    """
    支持快照隔离的 MVCC 内存 KV 存储
    
    核心特性：
    1. 快照隔离：每个读事务看到事务开始时刻的一致视图
    2. 读写不阻塞：读不需要加锁，写只在提交时短暂加锁
    3. 自动GC：基于低水位安全回收旧版本
    """
    
    def __init__(self, gc_enabled: bool = True, gc_interval: float = 0.1):
        self._txn_manager = TransactionManager()
        self._versions: Dict[str, List[Version]] = {}
        self._store_lock = threading.Lock()
        self._gc_enabled = gc_enabled
        self._gc_interval = gc_interval
        self._gc_running = False
        self._gc_thread: Optional[threading.Thread] = None
        self._gc_stats = {"collected_versions": 0, "gc_runs": 0}
        
        if gc_enabled:
            self._start_gc()
    
    # ============= 事务接口 =============
    
    def begin(self) -> Transaction:
        """开始一个新事务"""
        return self._txn_manager.begin_txn()
    
    def commit(self, txn: Transaction) -> bool:
        """
        提交事务
        
        对于写事务，需要：
        1. 分配 commit_ts
        2. 将暂存的写入应用到版本链（创建新版本）
        3. 标记旧版本过期
        4. 结束事务
        """
        if txn.status != "active":
            return False
        
        if txn.writes:
            commit_ts = self._txn_manager.allocate_ts()
            with self._store_lock:
                for key, value in txn.writes.items():
                    self._apply_write(key, value, commit_ts)
        
        self._txn_manager.end_txn(txn, "committed")
        return True
    
    def abort(self, txn: Transaction):
        """中止事务"""
        if txn.status == "active":
            self._txn_manager.end_txn(txn, "aborted")
    
    # ============= 读写操作 =============
    
    def get(self, txn: Transaction, key: str) -> Optional[Any]:
        """
        读取操作 - 基于快照的可见性判断
        
        可见性规则（快照隔离）：
        一个版本对事务 T 可见当且仅当：
        1. 版本的 create_ts <= T.read_ts   (在事务开始前已提交)
        2. 版本的 expire_ts > T.read_ts    (在事务开始前未被覆盖)
           或 expire_ts is None            (仍是最新版本)
        
        读操作不需要加锁，实现"读永不阻塞写"
        """
        if txn.status != "active":
            raise RuntimeError(f"Transaction is {txn.status}, cannot read")
        
        if key in txn.writes:
            return txn.writes[key]
        
        versions = self._versions.get(key, [])
        
        for version in versions:
            if version.create_ts <= txn.read_ts:
                if version.expire_ts is None or version.expire_ts > txn.read_ts:
                    return version.value
        
        return None
    
    def put(self, txn: Transaction, key: str, value: Any):
        """
        写入操作 - 暂存到事务本地，提交时才应用
        
        写操作在提交前不会修改全局版本链，实现"写永不阻塞读"
        """
        if txn.status != "active":
            raise RuntimeError(f"Transaction is {txn.status}, cannot write")
        txn.writes[key] = value
    
    def delete(self, txn: Transaction, key: str):
        """删除操作（通过写入 None 实现墓碑标记）"""
        self.put(txn, key, None)
    
    def _apply_write(self, key: str, value: Any, commit_ts: int):
        """
        提交时应用写入（必须在持有 _store_lock 时调用）
        
        版本链组织：
        - 按 create_ts 降序排列（最新版本在前）
        - 新版本插入链表头部
        - 旧版本的 expire_ts 设置为新版本的 create_ts
        """
        new_version = Version(value=value, create_ts=commit_ts)
        
        if key not in self._versions:
            self._versions[key] = [new_version]
        else:
            versions = self._versions[key]
            if versions:
                versions[0].expire_ts = commit_ts
            versions.insert(0, new_version)
    
    # ============= 垃圾回收 =============
    
    def _start_gc(self):
        """启动GC线程"""
        self._gc_running = True
        self._gc_thread = threading.Thread(target=self._gc_loop, daemon=True)
        self._gc_thread.start()
    
    def _gc_loop(self):
        """GC循环 - 定期清理过期版本"""
        while self._gc_running:
            try:
                self._collect_garbage()
            except Exception:
                pass
            time.sleep(self._gc_interval)
    
    def _collect_garbage(self) -> int:
        """
        垃圾回收核心逻辑
        
        回收时机的精确判断：
        一个版本可以被安全回收当且仅当：
        1. 它不是链表的第一个版本（即存在更新的版本）
        2. 它的 expire_ts <= low_water_mark
        
        为什么这两个条件足够？
        - 条件1：确保不是当前最新版本（最新版本可能被未来的新事务看到）
        - 条件2：expire_ts <= low_water_mark 意味着：
          所有可能看到这个版本的事务（read_ts >= create_ts 且 read_ts < expire_ts）
          都已经结束了。因为 low_water_mark 是所有活跃事务的最小 read_ts，
          如果 expire_ts <= low_water_mark，说明没有任何活跃事务的 read_ts
          落在 [create_ts, expire_ts) 区间内。
          （因为任何活跃事务的 read_ts >= low_water_mark >= expire_ts，
           所以 read_ts < expire_ts 不可能成立）
        
        回收策略：
        - 版本链按 create_ts 降序排列（新版本在前，旧版本在后）
        - 由于 expire_ts 也是单调递减的（旧版本先过期），一旦找到一个不可回收的版本，
          它后面的所有更旧版本也一定是不可回收的（wait，实际上是反过来）
        - 正确的逻辑：从前往后（从较新的版本开始）找第一个可回收的版本，
          然后删除它及之后的所有更旧版本（这些版本也都是可回收的）
        - 一次性截断链表尾部（O(1)操作）
        """
        low_water_mark = self._txn_manager.get_low_water_mark()
        collected = 0
        
        with self._store_lock:
            for key in list(self._versions.keys()):
                versions = self._versions[key]
                if len(versions) <= 1:
                    continue
                
                cutoff_idx = len(versions)
                for i in range(1, len(versions)):
                    version = versions[i]
                    if (version.expire_ts is not None and 
                        version.expire_ts <= low_water_mark):
                        cutoff_idx = i
                        break
                
                if cutoff_idx < len(versions):
                    collected += len(versions) - cutoff_idx
                    del versions[cutoff_idx:]
                
                if not versions:
                    del self._versions[key]
        
        if collected > 0:
            self._gc_stats["collected_versions"] += collected
            self._gc_stats["gc_runs"] += 1
        
        return collected
    
    def force_gc(self) -> int:
        """强制执行一次GC（用于测试）"""
        return self._collect_garbage()
    
    def stop_gc(self):
        """停止GC线程"""
        self._gc_running = False
        if self._gc_thread:
            self._gc_thread.join(timeout=1.0)
    
    # ============= 诊断和统计 =============
    
    def get_stats(self) -> Dict:
        """获取存储统计信息"""
        stats = {
            "total_keys": len(self._versions),
            "total_versions": sum(len(v) for v in self._versions.values()),
            "active_txns": self._txn_manager.get_active_txn_count(),
            "low_water_mark": self._txn_manager.get_low_water_mark(),
            "gc_stats": self._gc_stats.copy(),
            "current_timestamp": self._txn_manager._timestamp,
        }
        return stats
    
    def get_key_versions(self, key: str) -> List[Version]:
        """获取某个key的所有版本（用于调试）"""
        return self._versions.get(key, []).copy()
    
    def get_low_water_mark(self) -> int:
        """获取当前低水位"""
        return self._txn_manager.get_low_water_mark()
    
    def get_active_read_timestamps(self) -> Set[int]:
        """获取所有活跃事务的read_ts（用于调试）"""
        return self._txn_manager.get_active_read_timestamps()
    
    # ============= 便捷接口（自动事务） =============
    
    def simple_get(self, key: str) -> Optional[Any]:
        """便捷接口：单键读取（自动创建并提交事务）"""
        txn = self.begin()
        try:
            return self.get(txn, key)
        finally:
            self.commit(txn)
    
    def simple_put(self, key: str, value: Any):
        """便捷接口：单键写入（自动创建并提交事务）"""
        txn = self.begin()
        self.put(txn, key, value)
        self.commit(txn)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_gc()
