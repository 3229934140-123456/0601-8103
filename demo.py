import time
import threading
from mvcc_kv import MVCCKVStore


def demo_snapshot_isolation():
    """演示快照隔离"""
    print("=" * 60)
    print("演示1: 快照隔离 (Snapshot Isolation)")
    print("=" * 60)
    
    with MVCCKVStore(gc_enabled=False) as store:
        store.simple_put("account", 1000)
        print(f"初始余额: {store.simple_get('account')}")
        
        long_txn = store.begin()
        print(f"\n长事务开始，快照时间戳: {long_txn.read_ts}")
        print(f"长事务读取余额: {store.get(long_txn, 'account')}")
        
        print("\n--- 其他事务开始修改数据 ---")
        for i in range(5):
            txn = store.begin()
            current = store.get(txn, "account")
            store.put(txn, "account", current + 100)
            store.commit(txn)
            print(f"  事务{i+1}提交后余额: {store.simple_get('account')}")
        
        print(f"\n当前最新余额: {store.simple_get('account')}")
        print(f"长事务仍读取到: {store.get(long_txn, 'account')}")
        print("✓ 长事务看到的是一致的快照视图，不受后续写入影响")
        
        store.commit(long_txn)
        print("\n长事务提交完成")


def demo_version_chain():
    """演示版本链组织"""
    print("\n" + "=" * 60)
    print("演示2: MVCC版本链组织")
    print("=" * 60)
    
    with MVCCKVStore(gc_enabled=False) as store:
        for i in range(5):
            store.simple_put("key", f"value_{i}")
        
        versions = store.get_key_versions("key")
        print(f"\n'key'的版本链 (共{len(versions)}个版本):")
        print("-" * 60)
        print(f"{'索引':<6}{'值':<12}{'创建时间':<12}{'过期时间':<12}")
        print("-" * 60)
        for idx, v in enumerate(versions):
            expire = str(v.expire_ts) if v.expire_ts else "None(最新)"
            print(f"{idx:<6}{v.value:<12}{v.create_ts:<12}{expire:<12}")
        
        print("\n✓ 版本按创建时间降序排列（最新在前）")
        print("✓ 旧版本的expire_ts = 下一个新版本的create_ts")
        print("✓ 最新版本的expire_ts = None")


def demo_gc_with_long_txn():
    """演示长事务存在时的GC行为"""
    print("\n" + "=" * 60)
    print("演示3: 垃圾回收与长事务")
    print("=" * 60)
    
    with MVCCKVStore(gc_enabled=False) as store:
        store.simple_put("data", "version_0")
        print(f"初始数据: {store.simple_get('data')}")
        
        long_txn = store.begin()
        print(f"\n长事务开始，read_ts={long_txn.read_ts}")
        print(f"长事务读取: {store.get(long_txn, 'data')}")
        
        print("\n--- 写入多个新版本 ---")
        for i in range(1, 6):
            store.simple_put("data", f"version_{i}")
            print(f"  写入 version_{i}")
        
        versions = store.get_key_versions("data")
        print(f"\n当前版本数: {len(versions)}")
        print(f"活跃事务数: {store.get_stats()['active_txns']}")
        print(f"低水位 (Low Water Mark): {store.get_low_water_mark()}")
        
        collected = store.force_gc()
        print(f"\n尝试GC，回收了 {collected} 个版本")
        print(f"GC后版本数: {len(store.get_key_versions('data'))}")
        print("✓ 因为长事务还在运行，旧版本不能被回收")
        
        print(f"\n长事务仍能读取: {store.get(long_txn, 'data')}")
        
        store.commit(long_txn)
        print(f"\n长事务提交")
        print(f"新的低水位: {store.get_low_water_mark()}")
        
        collected = store.force_gc()
        print(f"\n再次GC，回收了 {collected} 个版本")
        print(f"GC后版本数: {len(store.get_key_versions('data'))}")
        print("✓ 长事务结束后，旧版本可以被安全回收")


def demo_concurrent_read_write():
    """演示读写不阻塞"""
    print("\n" + "=" * 60)
    print("演示4: 读写不阻塞 (Readers don't block writers)")
    print("=" * 60)
    
    with MVCCKVStore(gc_enabled=True, gc_interval=0.01) as store:
        store.simple_put("counter", 0)
        
        stop_flag = threading.Event()
        read_results = []
        write_count = [0]
        
        def reader():
            txn = store.begin()
            start_val = store.get(txn, "counter")
            for _ in range(10):
                val = store.get(txn, "counter")
                read_results.append(val)
                assert val == start_val, f"快照被破坏！期望{start_val}, 实际{val}"
                time.sleep(0.01)
            store.commit(txn)
            stop_flag.set()
        
        def writer():
            while not stop_flag.is_set():
                txn = store.begin()
                current = store.get(txn, "counter") or 0
                store.put(txn, "counter", current + 1)
                store.commit(txn)
                write_count[0] += 1
                time.sleep(0.005)
        
        print("\n启动读事务（运行0.1秒）和写事务（并发运行）...")
        
        reader_thread = threading.Thread(target=reader)
        writer_thread = threading.Thread(target=writer)
        
        reader_thread.start()
        writer_thread.start()
        
        reader_thread.join()
        writer_thread.join()
        
        print(f"读事务始终读取到一致的值: {read_results[0]}")
        print(f"写事务在期间完成了 {write_count[0]} 次写入")
        print(f"最终计数器值: {store.simple_get('counter')}")
        print("✓ 读事务的快照视图没有被并发写入破坏")
        print("✓ 读写完全并发，没有阻塞")


def demo_visibility_rules():
    """演示可见性判断规则"""
    print("\n" + "=" * 60)
    print("演示5: 可见性判断规则")
    print("=" * 60)
    
    with MVCCKVStore(gc_enabled=False) as store:
        print("\n可见性规则: 版本V对事务T可见 当且仅当")
        print("  1. V.create_ts <= T.read_ts  (在事务开始前已提交)")
        print("  2. V.expire_ts > T.read_ts   (在事务开始前未被覆盖)")
        print("     或 V.expire_ts is None    (仍是最新版本)")
        
        print("\n--- 场景演示 ---")
        store.simple_put("key", "v1")
        print("写入 v1")
        
        txn = store.begin()
        print(f"事务T开始，read_ts = {txn.read_ts}")
        
        store.simple_put("key", "v2")
        print("写入 v2")
        
        versions = store.get_key_versions("key")
        print(f"\n版本链:")
        for v in versions:
            expire = str(v.expire_ts) if v.expire_ts else "None"
            visible = (v.create_ts <= txn.read_ts and 
                      (v.expire_ts is None or v.expire_ts > txn.read_ts))
            status = "✓ 可见" if visible else "✗ 不可见"
            print(f"  {v.value}: create_ts={v.create_ts}, expire_ts={expire} → {status}")
        
        val = store.get(txn, "key")
        print(f"\n事务T实际读取到: {val}")
        print("✓ 事务T只能看到它开始前已提交且未被覆盖的版本")
        
        store.commit(txn)


if __name__ == "__main__":
    demo_snapshot_isolation()
    demo_version_chain()
    demo_gc_with_long_txn()
    demo_concurrent_read_write()
    demo_visibility_rules()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
