import threading
import time
import pytest
from mvcc_kv import MVCCKVStore, Version, Transaction


class TestMVCCBasics:
    """基础功能测试"""
    
    def test_simple_put_get(self):
        """测试基本的读写操作"""
        with MVCCKVStore(gc_enabled=False) as store:
            store.simple_put("key1", "value1")
            assert store.simple_get("key1") == "value1"
            assert store.simple_get("nonexistent") is None
    
    def test_transactional_put_get(self):
        """测试事务性读写"""
        with MVCCKVStore(gc_enabled=False) as store:
            txn = store.begin()
            store.put(txn, "key1", "value1")
            assert store.get(txn, "key1") == "value1"
            store.commit(txn)
            
            txn2 = store.begin()
            assert store.get(txn2, "key1") == "value1"
            store.commit(txn2)
    
    def test_write_not_visible_before_commit(self):
        """测试写入在提交前对其他事务不可见"""
        with MVCCKVStore(gc_enabled=False) as store:
            txn1 = store.begin()
            store.put(txn1, "key1", "value1")
            
            txn2 = store.begin()
            assert store.get(txn2, "key1") is None
            store.commit(txn2)
            
            store.commit(txn1)
            
            txn3 = store.begin()
            assert store.get(txn3, "key1") == "value1"
            store.commit(txn3)
    
    def test_abort_transaction(self):
        """测试事务中止"""
        with MVCCKVStore(gc_enabled=False) as store:
            txn = store.begin()
            store.put(txn, "key1", "value1")
            store.abort(txn)
            
            assert store.simple_get("key1") is None
    
    def test_read_own_writes(self):
        """测试事务能读取自己的写入"""
        with MVCCKVStore(gc_enabled=False) as store:
            txn = store.begin()
            store.put(txn, "key1", "value1")
            assert store.get(txn, "key1") == "value1"
            store.commit(txn)


class TestSnapshotIsolation:
    """快照隔离测试"""
    
    def test_snapshot_consistency(self):
        """测试快照一致性 - 长事务看到一致的视图"""
        with MVCCKVStore(gc_enabled=False) as store:
            store.simple_put("counter", 0)
            
            long_txn = store.begin()
            assert store.get(long_txn, "counter") == 0
            
            for i in range(5):
                txn = store.begin()
                store.put(txn, "counter", i + 1)
                store.commit(txn)
            
            assert store.simple_get("counter") == 5
            
            assert store.get(long_txn, "counter") == 0
            
            store.commit(long_txn)
    
    def test_multiple_concurrent_readers(self):
        """测试多个并发读事务各自看到一致的视图"""
        with MVCCKVStore(gc_enabled=False) as store:
            store.simple_put("x", 1)
            store.simple_put("y", 1)
            
            reader1 = store.begin()
            assert store.get(reader1, "x") == 1
            assert store.get(reader1, "y") == 1
            
            writer = store.begin()
            store.put(writer, "x", 2)
            store.put(writer, "y", 2)
            store.commit(writer)
            
            reader2 = store.begin()
            assert store.get(reader2, "x") == 2
            assert store.get(reader2, "y") == 2
            
            assert store.get(reader1, "x") == 1
            assert store.get(reader1, "y") == 1
            
            store.commit(reader1)
            store.commit(reader2)
    
    def test_repeatable_read(self):
        """测试可重复读"""
        with MVCCKVStore(gc_enabled=False) as store:
            store.simple_put("key", "v1")
            
            txn = store.begin()
            v1 = store.get(txn, "key")
            
            store.simple_put("key", "v2")
            
            v2 = store.get(txn, "key")
            
            assert v1 == v2 == "v1"
            store.commit(txn)


class TestVersionChain:
    """版本链组织测试"""
    
    def test_version_chain_creation(self):
        """测试版本链的创建和组织"""
        with MVCCKVStore(gc_enabled=False) as store:
            for i in range(5):
                store.simple_put("key", f"v{i}")
            
            versions = store.get_key_versions("key")
            assert len(versions) == 5
            
            assert versions[0].value == "v4"
            assert versions[0].expire_ts is None
            
            for i in range(1, 5):
                assert versions[i].expire_ts == versions[i - 1].create_ts
            
            assert versions[4].value == "v0"
    
    def test_version_timestamps_monotonic(self):
        """测试版本时间戳单调递增"""
        with MVCCKVStore(gc_enabled=False) as store:
            for i in range(10):
                store.simple_put("key", i)
            
            versions = store.get_key_versions("key")
            timestamps = [v.create_ts for v in versions]
            
            for i in range(len(timestamps) - 1):
                assert timestamps[i] > timestamps[i + 1]


class TestGarbageCollection:
    """垃圾回收测试 - 最核心的部分"""
    
    def test_gc_basic(self):
        """测试基本GC功能"""
        with MVCCKVStore(gc_enabled=False) as store:
            for i in range(10):
                store.simple_put("key", f"v{i}")
            
            assert len(store.get_key_versions("key")) == 10
            
            collected = store.force_gc()
            assert collected == 9
            assert len(store.get_key_versions("key")) == 1
    
    def test_gc_with_active_txn_preserves_old_versions(self):
        """
        测试有活跃事务时GC保留必要版本
        
        这是最棘手的场景：长事务正在运行时，GC不能回收它可能需要的版本
        """
        with MVCCKVStore(gc_enabled=False) as store:
            store.simple_put("key", "v0")
            
            long_txn = store.begin()
            assert store.get(long_txn, "key") == "v0"
            
            for i in range(1, 6):
                store.simple_put("key", f"v{i}")
            
            assert len(store.get_key_versions("key")) == 6
            
            collected = store.force_gc()
            assert collected == 0
            
            assert len(store.get_key_versions("key")) == 6
            
            assert store.get(long_txn, "key") == "v0"
            
            store.commit(long_txn)
            
            collected = store.force_gc()
            assert collected == 5
            assert len(store.get_key_versions("key")) == 1
    
    def test_gc_low_water_mark_tracking(self):
        """测试低水位追踪"""
        with MVCCKVStore(gc_enabled=False) as store:
            store.simple_put("key", "v0")
            
            assert store.get_low_water_mark() > 0
            
            txn1 = store.begin()
            
            store.simple_put("dummy", "1")
            txn2 = store.begin()
            
            store.simple_put("dummy", "2")
            txn3 = store.begin()
            
            lwm = store.get_low_water_mark()
            active_ts = store.get_active_read_timestamps()
            assert lwm == min(active_ts)
            assert len(active_ts) == 3, "三个事务应该有不同的read_ts"
            
            store.commit(txn1)
            
            new_lwm = store.get_low_water_mark()
            new_active_ts = store.get_active_read_timestamps()
            assert len(new_active_ts) == 2
            assert new_lwm > lwm, "t1结束后，低水位应该上升"
            
            store.commit(txn2)
            store.commit(txn3)
            
            final_lwm = store.get_low_water_mark()
            assert final_lwm >= new_lwm
    
    def test_gc_multiple_keys(self):
        """测试多key的GC"""
        with MVCCKVStore(gc_enabled=False) as store:
            for i in range(5):
                store.simple_put("key1", f"v{i}")
                store.simple_put("key2", f"v{i}")
            
            assert len(store.get_key_versions("key1")) == 5
            assert len(store.get_key_versions("key2")) == 5
            
            collected = store.force_gc()
            assert collected == 8
            
            assert len(store.get_key_versions("key1")) == 1
            assert len(store.get_key_versions("key2")) == 1
    
    def test_long_running_txn_blocks_gc(self):
        """
        长事务阻塞GC的精确时机测试
        
        时间轴：
        [阶段1] 提交 v0 → committed_ts = C0
        [阶段2] t1开始 → t1.read_ts = C0
        [阶段3] 提交 dummy写 → committed_ts = C1
        [阶段4] t2开始 → t2.read_ts = C1
        [阶段5] 提交 v1 → v0.expire_ts = C2, committed_ts = C2
        
        可见性分析：
        - t1 (read_ts=C0): 能看到v0 (create_ts=C0<=C0, expire_ts=C2>C0)
        - t2 (read_ts=C1): 能看到v0 (create_ts=C0<=C1, expire_ts=C2>C1)
                        看不到v1 (create_ts=C2>C1)
        
        GC时机分析：
        1. t1和t2都活跃时，低水位=min(C0, C1) = C0
           v0.expire_ts=C2 > C0 → 不可回收
        2. t1结束后，低水位=C1
           v0.expire_ts=C2 > C1 → 仍不可回收（t2还需要）
        3. t2结束后，低水位=C2（或更高）
           v0.expire_ts=C2 <= C2 → 可以回收！
        """
        with MVCCKVStore(gc_enabled=False) as store:
            txn_v0 = store.begin()
            store.put(txn_v0, "key", "v0")
            store.commit(txn_v0)
            
            versions = store.get_key_versions("key")
            v0_create_ts = versions[0].create_ts
            assert v0_create_ts > 0
            
            t1 = store.begin()
            t1_read_ts = t1.read_ts
            assert store.get(t1, "key") == "v0"
            
            store.simple_put("dummy", "1")
            
            t2 = store.begin()
            t2_read_ts = t2.read_ts
            assert store.get(t2, "key") == "v0"
            assert t2_read_ts > t1_read_ts, "t2应该在t1之后开始，有更大的read_ts"
            
            txn_v1 = store.begin()
            store.put(txn_v1, "key", "v1")
            store.commit(txn_v1)
            
            versions = store.get_key_versions("key")
            assert len(versions) == 2
            v1_create_ts = versions[0].create_ts
            v0_expire_ts = versions[1].expire_ts
            assert v0_expire_ts == v1_create_ts
            assert t2_read_ts < v0_expire_ts
            
            assert store.get(t2, "key") == "v0"
            assert store.get(t1, "key") == "v0"
            
            low_water = store.get_low_water_mark()
            assert low_water == min(t1_read_ts, t2_read_ts)
            
            collected = store.force_gc()
            assert collected == 0, "t1和t2都活跃时，v0不能被回收"
            
            store.commit(t1)
            
            low_water_after_t1 = store.get_low_water_mark()
            assert low_water_after_t1 == t2_read_ts
            
            collected = store.force_gc()
            assert collected == 0, "t1结束但t2还活跃时，v0仍不能被回收"
            
            assert store.get(t2, "key") == "v0", "t2仍应能看到v0"
            
            store.commit(t2)
            
            collected = store.force_gc()
            assert collected == 1, "t2结束后，v0可以被回收了"
            
            versions_final = store.get_key_versions("key")
            assert len(versions_final) == 1
            assert versions_final[0].value == "v1"


class TestConcurrency:
    """并发测试"""
    
    def test_concurrent_readers_writers(self):
        """测试并发读写不阻塞"""
        with MVCCKVStore(gc_enabled=True, gc_interval=0.01) as store:
            results = []
            errors = []
            
            reader_threads = []
            writer_threads = []
            
            def reader():
                try:
                    for _ in range(100):
                        txn = store.begin()
                        val = store.get(txn, "counter")
                        if val is not None:
                            results.append(val)
                        store.commit(txn)
                except Exception as e:
                    errors.append(e)
            
            def writer():
                try:
                    for i in range(50):
                        txn = store.begin()
                        current = store.get(txn, "counter") or 0
                        store.put(txn, "counter", current + 1)
                        store.commit(txn)
                except Exception as e:
                    errors.append(e)
            
            for _ in range(5):
                rt = threading.Thread(target=reader)
                wt = threading.Thread(target=writer)
                reader_threads.append(rt)
                writer_threads.append(wt)
                rt.start()
                wt.start()
            
            for rt in reader_threads:
                rt.join()
            for wt in writer_threads:
                wt.join()
            
            assert len(errors) == 0
            assert store.simple_get("counter") == 250
    
    def test_concurrent_snapshot_isolation(self):
        """测试并发下的快照隔离"""
        with MVCCKVStore(gc_enabled=False) as store:
            store.simple_put("x", 0)
            store.simple_put("y", 0)
            
            barrier = threading.Barrier(3)
            snapshots = []
            
            def long_reader():
                try:
                    txn = store.begin()
                    x1 = store.get(txn, "x")
                    y1 = store.get(txn, "y")
                    barrier.wait()
                    time.sleep(0.01)
                    x2 = store.get(txn, "x")
                    y2 = store.get(txn, "y")
                    store.commit(txn)
                    snapshots.append((x1, y1, x2, y2))
                except Exception as e:
                    print(f"Error in long_reader: {e}")
            
            def writer():
                try:
                    barrier.wait()
                    for i in range(10):
                        txn = store.begin()
                        store.put(txn, "x", i + 1)
                        store.put(txn, "y", i + 1)
                        store.commit(txn)
                except Exception as e:
                    print(f"Error in writer: {e}")
            
            t1 = threading.Thread(target=long_reader)
            t2 = threading.Thread(target=long_reader)
            t3 = threading.Thread(target=writer)
            
            t1.start()
            t2.start()
            t3.start()
            
            t1.join()
            t2.join()
            t3.join()
            
            assert len(snapshots) == 2
            for x1, y1, x2, y2 in snapshots:
                assert x1 == y1
                assert x2 == y2
                assert x1 == x2
                assert y1 == y2
    
    def test_gc_does_not_break_long_txn(self):
        """
        测试GC不会破坏长事务的快照视图"""
        with MVCCKVStore(gc_enabled=True, gc_interval=0.001) as store:
            store.simple_put("key", "initial")
            
            long_txn = store.begin()
            initial_val = store.get(long_txn, "key")
            
            gc_triggered = threading.Event()
            
            def writer_with_gc():
                try:
                    for i in range(100):
                        store.simple_put("key", f"v{i}")
                        time.sleep(0.001)
                    gc_triggered.set()
                except Exception as e:
                    print(f"Error in writer: {e}")
            
            writer_thread = threading.Thread(target=writer_with_gc)
            writer_thread.start()
            
            gc_triggered.wait()
            writer_thread.join()
            
            time.sleep(0.1)
            
            final_val = store.get(long_txn, "key")
            assert final_val == initial_val == "initial"
            
            store.commit(long_txn)
            
            time.sleep(0.1)
            store.force_gc()
            
            stats = store.get_stats()
            assert stats["gc_stats"]["gc_runs"] > 0
            assert stats["gc_stats"]["collected_versions"] > 0


class TestVisibilityRules:
    """可见性规则精确测试"""
    
    def test_visibility_create_ts(self):
        """测试create_ts <= read_ts的版本可见"""
        with MVCCKVStore(gc_enabled=False) as store:
            txn1 = store.begin()
            read_ts = txn1.read_ts
            
            txn2 = store.begin()
            store.put(txn2, "key", "value")
            store.commit(txn2)
            
            assert store.get(txn1, "key") is None
            store.commit(txn1)
            
            txn3 = store.begin()
            assert store.get(txn3, "key") == "value"
            store.commit(txn3)
    
    def test_visibility_expire_ts(self):
        """测试expire_ts > read_ts的版本可见"""
        with MVCCKVStore(gc_enabled=False) as store:
            store.simple_put("key", "v1")
            
            txn = store.begin()
            txn_read_ts = txn.read_ts
            
            store.simple_put("key", "v2")
            
            versions = store.get_key_versions("key")
            old_version = versions[1]
            assert old_version.expire_ts > txn_read_ts
            
            assert store.get(txn, "key") == "v1"
            store.commit(txn)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
