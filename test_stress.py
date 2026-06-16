"""
高并发稳定性测试

包含三个核心测试场景：
1. 快照一致性压测：多轮重复，验证读事务始终看到一致的批量视图
2. 写写冲突压测：验证并发写入不丢失更新，结果与成功次数对齐
3. 综合压测：快照一致性 + 冲突检测 + GC 一起跑，验证长事务不被破坏
"""

import threading
import time
import random
import pytest
from mvcc_kv import MVCCKVStore, Transaction


NUM_ROUNDS = 20
NUM_THREADS = 10
OPS_PER_THREAD = 50


def _increment_with_retry(store: MVCCKVStore, key: str, max_retries: int = 100) -> bool:
    """
    带自动重试的原子递增操作
    
    使用OCC（乐观并发控制）+ 重试的模式
    返回True表示最终成功，False表示重试次数耗尽
    """
    for attempt in range(max_retries):
        txn = store.begin()
        current = store.get(txn, key) or 0
        store.put(txn, key, current + 1)
        if store.commit(txn):
            return True
    return False


class TestSnapshotConsistencyStress:
    """
    快照一致性压力测试
    
    验证：一个事务同时改x和y时，并发读事务看到的始终是同一批次的数据
    要么都是提交前的值，要么都是提交后的值，不会出现中间状态
    """
    
    def test_snapshot_consistency_multi_round(self):
        """
        多轮快照一致性压测
        
        场景：
        - 多个写事务，每个事务同时修改x和y（保证原子提交）
        - 多个读事务，每个事务读取x和y，验证x == y
        - 重复多轮，确保不会因线程调度差异而偶然通过
        """
        for round_num in range(NUM_ROUNDS):
            self._run_snapshot_round(round_num)
    
    def _run_snapshot_round(self, round_num: int):
        """运行一轮快照一致性测试"""
        with MVCCKVStore(gc_enabled=False) as store:
            txn = store.begin()
            store.put(txn, "x", 0)
            store.put(txn, "y", 0)
            store.commit(txn)
            
            inconsistencies = []
            errors = []
            barrier = threading.Barrier(NUM_THREADS * 2)
            
            def writer():
                try:
                    barrier.wait()
                    for i in range(OPS_PER_THREAD):
                        success = False
                        while not success:
                            txn = store.begin()
                            x = store.get(txn, "x") or 0
                            y = store.get(txn, "y") or 0
                            store.put(txn, "x", x + 1)
                            store.put(txn, "y", y + 1)
                            success = store.commit(txn)
                except Exception as e:
                    errors.append(e)
            
            def reader():
                try:
                    barrier.wait()
                    for i in range(OPS_PER_THREAD):
                        txn = store.begin()
                        x = store.get(txn, "x")
                        y = store.get(txn, "y")
                        if x != y:
                            inconsistencies.append((x, y, txn.read_ts))
                        store.commit(txn)
                except Exception as e:
                    errors.append(e)
            
            threads = []
            for _ in range(NUM_THREADS):
                threads.append(threading.Thread(target=writer))
                threads.append(threading.Thread(target=reader))
            
            for t in threads:
                t.start()
            
            for t in threads:
                t.join()
            
            assert len(errors) == 0, f"第{round_num}轮出现错误: {errors}"
            assert len(inconsistencies) == 0, (
                f"第{round_num}轮发现 {len(inconsistencies)} 次快照不一致！\n"
                f"前3个不一致: {inconsistencies[:3]}\n"
                f"x和y应该始终相等（要么都是旧值，要么都是新值）"
            )
            
            final_x = store.simple_get("x")
            final_y = store.simple_get("y")
            assert final_x == final_y, f"最终x={final_x}, y={final_y}，不相等"
            assert final_x == NUM_THREADS * OPS_PER_THREAD, (
                f"最终值={final_x}, 期望={NUM_THREADS * OPS_PER_THREAD}"
            )


class TestWriteWriteConflictStress:
    """
    写写冲突压力测试
    
    验证：
    1. 两个事务都从counter=0开始做加1并提交时，不会静默丢掉更新
    2. 同一个key被并发写入时，后提交的一方会明确失败
    3. 最终结果和提交成功次数能对上
    """
    
    def test_counter_increment_no_lost_update(self):
        """
        计数器递增：验证不丢失更新
        
        每个线程做OPS_PER_THREAD次递增，每次递增都带重试
        最终结果应该等于 线程数 × 每线程操作数
        """
        for round_num in range(NUM_ROUNDS):
            self._run_counter_round(round_num)
    
    def _run_counter_round(self, round_num: int):
        """运行一轮计数器递增测试"""
        with MVCCKVStore(gc_enabled=False) as store:
            store.simple_put("counter", 0)
            
            total_success = [0]
            total_failures = [0]
            errors = []
            lock = threading.Lock()
            barrier = threading.Barrier(NUM_THREADS)
            
            def worker():
                try:
                    barrier.wait()
                    success_count = 0
                    for i in range(OPS_PER_THREAD):
                        if _increment_with_retry(store, "counter", max_retries=200):
                            success_count += 1
                        else:
                            with lock:
                                total_failures[0] += 1
                    with lock:
                        total_success[0] += success_count
                except Exception as e:
                    errors.append(e)
            
            threads = []
            for _ in range(NUM_THREADS):
                threads.append(threading.Thread(target=worker))
            
            for t in threads:
                t.start()
            
            for t in threads:
                t.join()
            
            assert len(errors) == 0, f"第{round_num}轮出现错误: {errors}"
            
            final_value = store.simple_get("counter")
            expected = NUM_THREADS * OPS_PER_THREAD
            
            assert final_value == expected, (
                f"第{round_num}轮失败！\n"
                f"最终counter值: {final_value}\n"
                f"期望: {expected}\n"
                f"成功提交次数: {total_success[0]}\n"
                f"失败次数（重试耗尽）: {total_failures[0]}\n"
                f"结果与成功次数对齐检查: {final_value} == {total_success[0]} ? {final_value == total_success[0]}"
            )
            
            assert final_value == total_success[0], (
                f"第{round_num}轮：最终值({final_value})与成功次数({total_success[0]})对不上！"
            )
    
    def test_conflict_detection_is_deterministic(self):
        """
        验证冲突检测是确定性的：首提交者胜出
        
        用两个线程交错执行，精确控制时序来验证
        """
        for round_num in range(NUM_ROUNDS):
            with MVCCKVStore(gc_enabled=False) as store:
                store.simple_put("key", 0)
                
                txn1 = store.begin()
                v1 = store.get(txn1, "key")
                
                txn2 = store.begin()
                v2 = store.get(txn2, "key")
                
                assert v1 == v2 == 0
                
                store.put(txn1, "key", v1 + 1)
                result1 = store.commit(txn1)
                
                store.put(txn2, "key", v2 + 1)
                result2 = store.commit(txn2)
                
                assert result1 == True, "首提交者应该成功"
                assert result2 == False, "后提交者应该失败（写-写冲突）"
                
                final_value = store.simple_get("key")
                assert final_value == 1, f"最终值应为1（只有第一个提交成功），实际是{final_value}"


class TestIntegratedStress:
    """
    综合压力测试
    
    同时运行：
    - 长事务（验证快照一致性，且不被GC破坏）
    - 多个写事务（带冲突检测和重试）
    - 多个读事务（验证快照一致性）
    - GC线程（自动回收旧版本）
    
    验证所有组件在一起工作时的正确性
    """
    
    def test_integrated_with_long_txn(self):
        """
        综合测试：长事务 + 并发读写 + GC
        
        验证：
        1. 长事务始终能看到一致的快照（不被并发写入影响）
        2. GC不会回收长事务还需要的版本
        3. 长事务结束后，GC能正常回收旧版本
        """
        for round_num in range(5):
            self._run_integrated_round(round_num)
    
    def _run_integrated_round(self, round_num: int):
        """运行一轮综合测试"""
        with MVCCKVStore(gc_enabled=True, gc_interval=0.005) as store:
            store.simple_put("counter", 0)
            store.simple_put("x", 0)
            store.simple_put("y", 0)
            
            long_txn = store.begin()
            long_initial_counter = store.get(long_txn, "counter")
            long_initial_x = store.get(long_txn, "x")
            long_initial_y = store.get(long_txn, "y")
            
            assert long_initial_counter == 0
            assert long_initial_x == 0
            assert long_initial_y == 0
            
            stop_flag = threading.Event()
            errors = []
            read_inconsistencies = []
            write_success = [0]
            lock = threading.Lock()
            
            def writer_worker():
                try:
                    while not stop_flag.is_set():
                        txn = store.begin()
                        c = store.get(txn, "counter") or 0
                        x = store.get(txn, "x") or 0
                        y = store.get(txn, "y") or 0
                        store.put(txn, "counter", c + 1)
                        store.put(txn, "x", x + 1)
                        store.put(txn, "y", y + 1)
                        if store.commit(txn):
                            with lock:
                                write_success[0] += 1
                except Exception as e:
                    errors.append(("writer", e))
            
            def reader_worker():
                try:
                    while not stop_flag.is_set():
                        txn = store.begin()
                        x = store.get(txn, "x")
                        y = store.get(txn, "y")
                        c = store.get(txn, "counter")
                        if x != y or x != c:
                            read_inconsistencies.append((x, y, c))
                        store.commit(txn)
                except Exception as e:
                    errors.append(("reader", e))
            
            writer_threads = []
            reader_threads = []
            
            for _ in range(4):
                t = threading.Thread(target=writer_worker)
                writer_threads.append(t)
                t.start()
            
            for _ in range(4):
                t = threading.Thread(target=reader_worker)
                reader_threads.append(t)
                t.start()
            
            time.sleep(0.3)
            
            long_counter = store.get(long_txn, "counter")
            long_x = store.get(long_txn, "x")
            long_y = store.get(long_txn, "y")
            
            assert long_counter == long_initial_counter, (
                f"长事务的counter值变了！初始={long_initial_counter}, 现在={long_counter}"
            )
            assert long_x == long_initial_x, (
                f"长事务的x值变了！初始={long_initial_x}, 现在={long_x}"
            )
            assert long_y == long_initial_y, (
                f"长事务的y值变了！初始={long_initial_y}, 现在={long_y}"
            )
            
            assert long_x == long_y == long_counter, (
                f"长事务看到的x,y,c不一致！x={long_x}, y={long_y}, c={long_counter}"
            )
            
            stats = store.get_stats()
            assert stats["gc_stats"]["gc_runs"] >= 0
            
            stop_flag.set()
            
            for t in writer_threads:
                t.join()
            for t in reader_threads:
                t.join()
            
            assert len(errors) == 0, f"第{round_num}轮出现错误: {errors}"
            assert len(read_inconsistencies) == 0, (
                f"第{round_num}轮发现 {len(read_inconsistencies)} 次读不一致"
            )
            
            store.commit(long_txn)
            
            time.sleep(0.1)
            store.force_gc()
            
            final_stats = store.get_stats()
            assert final_stats["gc_stats"]["collected_versions"] > 0, (
                "长事务结束后应该能回收一些旧版本"
            )
            
            final_counter = store.simple_get("counter")
            final_x = store.simple_get("x")
            final_y = store.simple_get("y")
            
            assert final_counter == final_x == final_y, (
                f"最终值不一致：counter={final_counter}, x={final_x}, y={final_y}"
            )
            assert final_counter == write_success[0], (
                f"最终counter值({final_counter})与成功写入次数({write_success[0]})对不上"
            )
    
    def test_many_keys_gc_with_long_txn(self):
        """
        多key场景下的GC正确性验证
        
        大量key被频繁修改，同时有长事务在运行
        验证GC不会误删长事务需要的版本
        """
        with MVCCKVStore(gc_enabled=True, gc_interval=0.001) as store:
            NUM_KEYS = 100
            
            for i in range(NUM_KEYS):
                store.simple_put(f"key_{i}", 0)
            
            long_txn = store.begin()
            initial_values = {}
            for i in range(NUM_KEYS):
                initial_values[f"key_{i}"] = store.get(long_txn, f"key_{i}")
            
            stop_flag = threading.Event()
            errors = []
            
            def writer():
                try:
                    while not stop_flag.is_set():
                        key_idx = random.randint(0, NUM_KEYS - 1)
                        key = f"key_{key_idx}"
                        _increment_with_retry(store, key, max_retries=50)
                except Exception as e:
                    errors.append(e)
            
            writer_threads = []
            for _ in range(5):
                t = threading.Thread(target=writer)
                writer_threads.append(t)
                t.start()
            
            time.sleep(0.5)
            
            for i in range(NUM_KEYS):
                key = f"key_{i}"
                val = store.get(long_txn, key)
                assert val == initial_values[key], (
                    f"长事务看到的{key}变了！初始={initial_values[key]}, 现在={val}"
                )
            
            stop_flag.set()
            for t in writer_threads:
                t.join()
            
            assert len(errors) == 0, f"出现错误: {errors}"
            
            store.commit(long_txn)
            
            time.sleep(0.1)
            collected = store.force_gc()
            
            stats = store.get_stats()
            assert stats["total_keys"] == NUM_KEYS
            
            for i in range(NUM_KEYS):
                versions = store.get_key_versions(f"key_{i}")
                assert len(versions) >= 1, f"key_{i}没有版本了！"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
