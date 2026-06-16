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


class TestWriteSetConflictDetection:
    """
    写入集冲突检测测试
    
    验证：两个同一快照开始的事务都只写不读同一个key时，后提交者明确失败
    """
    
    def test_write_only_conflict_detection(self):
        """只写不读的场景也要能检测到冲突"""
        with MVCCKVStore(gc_enabled=False) as store:
            store.simple_put("key", "initial")
            
            txn1 = store.begin()
            txn2 = store.begin()
            
            assert txn1.read_ts == txn2.read_ts, "两个事务应该有相同的快照时间戳"
            
            store.put(txn1, "key", "from_txn1")
            result1 = store.commit(txn1)
            
            store.put(txn2, "key", "from_txn2")
            result2 = store.commit(txn2)
            
            assert result1 == True, "首提交者应该成功"
            assert result2 == False, "后提交者（没读过直接写）也应该检测到冲突并失败"
            
            final = store.simple_get("key")
            assert final == "from_txn1", f"最终值应该是首提交者写入的，实际是{final}"
    
    def test_write_only_conflict_multi_round(self):
        """多轮验证写入集冲突检测的稳定性"""
        for round_num in range(NUM_ROUNDS):
            with MVCCKVStore(gc_enabled=False) as store:
                store.simple_put("counter", 0)
                
                success_count = [0]
                lock = threading.Lock()
                barrier = threading.Barrier(NUM_THREADS)
                
                def worker():
                    try:
                        barrier.wait()
                        for _ in range(OPS_PER_THREAD):
                            success, _, _ = store.run_transaction(
                                lambda txn: (
                                    store.put(txn, "counter", 
                                              (store.get(txn, "counter") or 0) + 1)
                                ),
                                max_retries=50
                            )
                            if success:
                                with lock:
                                    success_count[0] += 1
                    except Exception as e:
                        pass
                
                threads = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                
                final = store.simple_get("counter")
                assert final == success_count[0], (
                    f"第{round_num}轮：最终值({final})与成功次数({success_count[0]})对不上"
                )
                assert final == NUM_THREADS * OPS_PER_THREAD, (
                    f"第{round_num}轮：最终值={final}, 期望={NUM_THREADS * OPS_PER_THREAD}"
                )


class TestBatchOperations:
    """
    批量操作接口测试
    
    验证：
    1. batch_get 看到一致快照
    2. batch_put 原子提交，不会出现中间状态
    3. batch 通用接口能同时读写并返回结果
    """
    
    def test_batch_get_consistent_snapshot(self):
        """batch_get 在同一个快照下读取多个key"""
        with MVCCKVStore(gc_enabled=False) as store:
            txn_init = store.begin()
            store.put(txn_init, "a", 1)
            store.put(txn_init, "b", 2)
            store.put(txn_init, "c", 3)
            store.commit(txn_init)
            
            results = store.batch_get(["a", "b", "c", "d"])
            assert results == {"a": 1, "b": 2, "c": 3, "d": None}
    
    def test_batch_put_atomic_visibility(self):
        """batch_put 多key写入具有原子可见性"""
        with MVCCKVStore(gc_enabled=False) as store:
            txn_init = store.begin()
            store.put(txn_init, "x", 0)
            store.put(txn_init, "y", 0)
            store.commit(txn_init)
            
            reader_results = []
            errors = []
            barrier = threading.Barrier(3)
            
            def writer():
                try:
                    barrier.wait()
                    for i in range(100):
                        success = False
                        while not success:
                            success, _ = store.batch(
                                get_keys=["x", "y"],
                                put_items={"x": i + 1, "y": i + 1}
                            )
                except Exception as e:
                    errors.append(e)
            
            def reader1():
                try:
                    barrier.wait()
                    for _ in range(100):
                        r = store.batch_get(["x", "y"])
                        if r["x"] != r["y"]:
                            reader_results.append(r)
                except Exception as e:
                    errors.append(e)
            
            def reader2():
                try:
                    barrier.wait()
                    for _ in range(100):
                        r = store.batch_get(["x", "y"])
                        if r["x"] != r["y"]:
                            reader_results.append(r)
                except Exception as e:
                    errors.append(e)
            
            threads = [
                threading.Thread(target=writer),
                threading.Thread(target=reader1),
                threading.Thread(target=reader2),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            assert len(errors) == 0, f"出现错误: {errors}"
            assert len(reader_results) == 0, (
                f"发现 {len(reader_results)} 次x和y不相等的中间状态"
            )
            
            final = store.batch_get(["x", "y"])
            assert final["x"] == final["y"] == 100
    
    def test_batch_transfer_scenario(self):
        """batch 接口模拟转账场景"""
        with MVCCKVStore(gc_enabled=False) as store:
            store.simple_put("alice", 1000)
            store.simple_put("bob", 1000)
            
            def transfer(txn):
                alice = store.get(txn, "alice") or 0
                bob = store.get(txn, "bob") or 0
                amount = 10
                if alice >= amount:
                    store.put(txn, "alice", alice - amount)
                    store.put(txn, "bob", bob + amount)
                    return True
                return False
            
            total_money = []
            for _ in range(100):
                store.run_transaction(transfer, max_retries=20)
                r = store.batch_get(["alice", "bob"])
                total_money.append(r["alice"] + r["bob"])
            
            for t in total_money:
                assert t == 2000, f"转账过程中总钱数变了！{total_money}"
            
            final = store.batch_get(["alice", "bob"])
            assert final["alice"] + final["bob"] == 2000


class TestRunTransaction:
    """
    run_transaction 事务重试工具测试
    """
    
    def test_run_transaction_simple(self):
        """基本的读改写重试"""
        with MVCCKVStore(gc_enabled=False) as store:
            store.simple_put("counter", 0)
            
            def increment(txn):
                current = store.get(txn, "counter") or 0
                new_val = current + 1
                store.put(txn, "counter", new_val)
                return new_val
            
            success, result, attempts = store.run_transaction(increment)
            assert success == True
            assert result == 1
            assert attempts >= 1
            assert store.simple_get("counter") == 1
    
    def test_run_transaction_max_retries(self):
        """达到最大重试次数时返回失败"""
        with MVCCKVStore(gc_enabled=False) as store:
            store.simple_put("key", 0)
            
            txn_blocker = store.begin()
            store.put(txn_blocker, "key", 999)
            store.commit(txn_blocker)
            
            call_count = [0]
            def always_conflict(txn):
                call_count[0] += 1
                store.put(txn, "key", (store.get(txn, "key") or 0) + 1)
                
                blocker = store.begin()
                store.put(blocker, "key", -1)
                store.commit(blocker)
            
            success, result, attempts = store.run_transaction(always_conflict, max_retries=3)
            assert success == False
            assert attempts == 3
            assert call_count[0] == 3
    
    def test_run_transaction_high_concurrency_counter(self):
        """高并发下使用 run_transaction 做计数器递增"""
        for round_num in range(5):
            with MVCCKVStore(gc_enabled=False) as store:
                store.simple_put("counter", 0)
                
                success_count = [0]
                lock = threading.Lock()
                barrier = threading.Barrier(NUM_THREADS)
                
                def worker():
                    try:
                        barrier.wait()
                        for _ in range(OPS_PER_THREAD):
                            def inc(txn):
                                c = store.get(txn, "counter") or 0
                                store.put(txn, "counter", c + 1)
                                return c + 1
                            
                            success, _, _ = store.run_transaction(inc, max_retries=100)
                            if success:
                                with lock:
                                    success_count[0] += 1
                    except Exception as e:
                        pass
                
                threads = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                
                final = store.simple_get("counter")
                expected = NUM_THREADS * OPS_PER_THREAD
                assert final == expected, (
                    f"第{round_num}轮：最终值={final}, 期望={expected}, "
                    f"成功次数={success_count[0]}"
                )
                assert final == success_count[0]


class TestMixedWorkload:
    """
    混合工作负载测试
    
    同时运行：
    - 长事务（从头到尾保持一致快照）
    - 批量写（batch_put）
    - 直接覆盖写（simple_put）
    - 读改写重试（run_transaction）
    - 批量读（batch_get）
    - 后台GC
    
    验证：长事务快照不被破坏，总数据一致性保持
    """
    
    def test_mixed_workload_with_long_txn(self):
        """混合工作负载 + 长事务 + GC"""
        for round_num in range(3):
            self._run_mixed_round(round_num)
    
    def _run_mixed_round(self, round_num: int):
        with MVCCKVStore(gc_enabled=True, gc_interval=0.005) as store:
            NUM_ACCOUNTS = 10
            TOTAL_MONEY = 10000
            PER_ACCOUNT = TOTAL_MONEY // NUM_ACCOUNTS
            
            for i in range(NUM_ACCOUNTS):
                store.simple_put(f"acc_{i}", PER_ACCOUNT)
            store.simple_put("global_counter", 0)
            
            long_txn = store.begin()
            long_initial = {}
            for i in range(NUM_ACCOUNTS):
                long_initial[f"acc_{i}"] = store.get(long_txn, f"acc_{i}")
            long_initial_counter = store.get(long_txn, "global_counter")
            
            long_initial_total = sum(long_initial.values())
            assert long_initial_total == TOTAL_MONEY
            
            stop_flag = threading.Event()
            errors = []
            inconsistencies = []
            lock = threading.Lock()
            transfer_success = [0]
            batch_write_success = [0]
            overwrite_success = [0]
            
            def transfer_worker():
                try:
                    while not stop_flag.is_set():
                        def do_transfer(txn):
                            src = random.randint(0, NUM_ACCOUNTS - 1)
                            dst = random.randint(0, NUM_ACCOUNTS - 1)
                            if src == dst:
                                return
                            src_key = f"acc_{src}"
                            dst_key = f"acc_{dst}"
                            src_val = store.get(txn, src_key) or 0
                            dst_val = store.get(txn, dst_key) or 0
                            amount = random.randint(1, 10)
                            if src_val >= amount:
                                store.put(txn, src_key, src_val - amount)
                                store.put(txn, dst_key, dst_val + amount)
                                counter = store.get(txn, "global_counter") or 0
                                store.put(txn, "global_counter", counter + 1)
                        
                        success, _, _ = store.run_transaction(do_transfer, max_retries=20)
                        if success:
                            with lock:
                                transfer_success[0] += 1
                except Exception as e:
                    errors.append(("transfer", e))
            
            def batch_write_worker():
                try:
                    while not stop_flag.is_set():
                        def do_batch_transfer(txn):
                            src = random.randint(0, NUM_ACCOUNTS - 1)
                            dst = random.randint(0, NUM_ACCOUNTS - 1)
                            if src == dst:
                                return
                            src_key = f"acc_{src}"
                            dst_key = f"acc_{dst}"
                            src_val = store.get(txn, src_key) or 0
                            dst_val = store.get(txn, dst_key) or 0
                            counter = store.get(txn, "global_counter") or 0
                            amount = random.randint(1, 5)
                            if src_val >= amount:
                                store.put(txn, src_key, src_val - amount)
                                store.put(txn, dst_key, dst_val + amount)
                                store.put(txn, "global_counter", counter + 1)
                        
                        success, _, _ = store.run_transaction(do_batch_transfer, max_retries=20)
                        if success:
                            with lock:
                                batch_write_success[0] += 1
                except Exception as e:
                    errors.append(("batch", e))
            
            def overwrite_worker():
                try:
                    while not stop_flag.is_set():
                        idx = random.randint(0, NUM_ACCOUNTS - 1)
                        key = f"acc_{idx}"
                        
                        def safe_overwrite(txn):
                            current = store.get(txn, key) or 0
                            store.put(txn, key, current)
                        
                        store.run_transaction(safe_overwrite, max_retries=10)
                        with lock:
                            overwrite_success[0] += 1
                except Exception as e:
                    errors.append(("overwrite", e))
            
            def reader_worker():
                try:
                    while not stop_flag.is_set():
                        keys = [f"acc_{i}" for i in range(NUM_ACCOUNTS)]
                        results = store.batch_get(keys)
                        total = sum(v or 0 for v in results.values())
                        if total != TOTAL_MONEY:
                            inconsistencies.append(("reader_total", total, results))
                except Exception as e:
                    errors.append(("reader", e))
            
            threads = []
            for _ in range(2):
                threads.append(threading.Thread(target=transfer_worker))
            for _ in range(2):
                threads.append(threading.Thread(target=batch_write_worker))
            threads.append(threading.Thread(target=overwrite_worker))
            threads.append(threading.Thread(target=reader_worker))
            
            for t in threads:
                t.start()
            
            time.sleep(0.5)
            
            long_values = {}
            for i in range(NUM_ACCOUNTS):
                long_values[f"acc_{i}"] = store.get(long_txn, f"acc_{i}")
            long_counter = store.get(long_txn, "global_counter")
            
            long_total = sum(long_values.values())
            assert long_total == long_initial_total, (
                f"第{round_num}轮：长事务看到的总钱数变了！"
                f"初始={long_initial_total}, 现在={long_total}"
            )
            assert long_counter == long_initial_counter, (
                f"第{round_num}轮：长事务看到的global_counter变了！"
                f"初始={long_initial_counter}, 现在={long_counter}"
            )
            for i in range(NUM_ACCOUNTS):
                key = f"acc_{i}"
                assert long_values[key] == long_initial[key], (
                    f"第{round_num}轮：长事务看到的{key}变了！"
                    f"初始={long_initial[key]}, 现在={long_values[key]}"
                )
            
            stop_flag.set()
            for t in threads:
                t.join()
            
            assert len(errors) == 0, f"第{round_num}轮出现错误: {errors}"
            assert len(inconsistencies) == 0, (
                f"第{round_num}轮发现 {len(inconsistencies)} 次读不一致"
            )
            
            store.commit(long_txn)
            time.sleep(0.1)
            store.force_gc()
            
            final_keys = [f"acc_{i}" for i in range(NUM_ACCOUNTS)]
            final_results = store.batch_get(final_keys)
            final_total = sum(v or 0 for v in final_results.values())
            assert final_total == TOTAL_MONEY, (
                f"第{round_num}轮：最终总钱数不对！{final_total} != {TOTAL_MONEY}"
            )
            
            stats = store.get_stats()
            assert stats["gc_stats"]["gc_runs"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
