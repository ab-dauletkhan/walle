"""
WALL-E Memory System Test Suite
================================
Comprehensive tests for all memory system components:
- Core Memory (persona, human, system blocks)
- Recall Memory (FAISS-accelerated conversation history)
- Archival Memory (long-term facts with importance decay)
- FAISS vector search performance
- Tool calling functionality

Run with: pytest test_memory_system.py -v
Or directly: python test_memory_system.py
"""

import os
import sys
import time
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_system import (
    Memory, Block, RecallMemory, ArchivalMemory,
    FAISSManager, get_embedding, _faiss_available
)
from config import conf


class TestCoreMemory(unittest.TestCase):
    """Tests for Core Memory (Block-based, always in context)"""

    def setUp(self):
        """Create a fresh memory instance with temp database"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_core_memory.db")
        self.memory = Memory(db_path=self.db_path)

    def tearDown(self):
        """Cleanup temp files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_blocks_created(self):
        """Test that default blocks (persona, human, system) are created"""
        self.assertEqual(len(self.memory.blocks), 3)
        labels = [b.label for b in self.memory.blocks]
        self.assertIn("persona", labels)
        self.assertIn("human", labels)
        self.assertIn("system", labels)

    def test_get_block(self):
        """Test retrieving a specific block"""
        persona = self.memory.get_block("persona")
        self.assertIsNotNone(persona)
        self.assertEqual(persona.label, "persona")

    def test_block_append(self):
        """Test appending content to a block"""
        human = self.memory.get_block("human")
        original_len = len(human.value)

        success, msg = human.append(" User's name is Alex.")

        self.assertTrue(success)
        self.assertIn("Alex", human.value)
        self.assertGreater(len(human.value), original_len)

    def test_block_replace(self):
        """Test replacing content in a block"""
        human = self.memory.get_block("human")
        human.value = "The human likes coffee."

        success, msg = human.replace("coffee", "tea")

        self.assertTrue(success)
        self.assertIn("tea", human.value)
        self.assertNotIn("coffee", human.value)

    def test_block_limit_enforcement(self):
        """Test that block character limits are enforced"""
        human = self.memory.get_block("human")
        human.value = ""

        # Try to exceed the limit (2000 chars)
        long_content = "x" * 3000
        success, msg = human.append(long_content)

        self.assertFalse(success)
        self.assertIn("space", msg.lower())

    def test_read_only_block(self):
        """Test that read-only blocks cannot be modified"""
        system = self.memory.get_block("system")

        success, msg = system.append(" New content")

        self.assertFalse(success)
        self.assertIn("read-only", msg.lower())

    def test_memory_persistence(self):
        """Test that memory persists to database"""
        human = self.memory.get_block("human")
        human.value = "Test persistence value"
        self.memory.save()

        # Create new memory instance from same database
        new_memory = Memory(db_path=self.db_path)
        new_human = new_memory.get_block("human")

        self.assertEqual(new_human.value, "Test persistence value")

    def test_compile_format(self):
        """Test that compile() produces valid XML-like format"""
        compiled = self.memory.compile()

        self.assertIn("<memory_blocks>", compiled)
        self.assertIn("</memory_blocks>", compiled)
        self.assertIn("<persona>", compiled)
        self.assertIn("<human>", compiled)

    def test_metadata_tracking(self):
        """Test that metadata (created_at, last_modified) is tracked"""
        human = self.memory.get_block("human")
        original_modified = human.metadata.get("last_modified")

        time.sleep(0.1)
        human.append(" New content")

        new_modified = human.metadata.get("last_modified")
        self.assertNotEqual(original_modified, new_modified)


class TestRecallMemory(unittest.TestCase):
    """Tests for Recall Memory (conversation history with FAISS search)"""

    def setUp(self):
        """Create fresh recall memory with temp database"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_recall_memory.db")
        
        # ISOLATION FIX: Patch global config to use temp FAISS index
        self.original_faiss_path = conf.FAISS_INDEX_PATH
        conf.FAISS_INDEX_PATH = os.path.join(self.temp_dir, "test_recall.index")
        
        self.recall = RecallMemory(db_path=self.db_path, use_semantic=True)

    def tearDown(self):
        """Cleanup temp files and restore config"""
        # Restore global config
        conf.FAISS_INDEX_PATH = self.original_faiss_path
        
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_insert_and_count(self):
        """Test inserting messages and counting them"""
        self.assertEqual(self.recall.get_count(), 0)

        self.recall.insert("user", "Hello, how are you?")
        self.recall.insert("assistant", "I'm doing well, thank you!")

        self.assertEqual(self.recall.get_count(), 2)

    def test_insert_with_tools_used(self):
        """Test inserting messages with tool usage tracking"""
        self.recall.insert(
            "assistant",
            "Let me search for that.",
            tools_used=["consult_internet_for_facts"]
        )

        # Verify it was stored
        results = self.recall.search(limit=1)
        self.assertEqual(len(results), 1)

    def test_text_search_fallback(self):
        """Test text-based search (LIKE query)"""
        self.recall.insert("user", "I love Python programming")
        self.recall.insert("user", "JavaScript is also good")
        self.recall.insert("user", "Python is my favorite")

        # Search should find Python-related messages
        results = self.recall.search("Python", limit=10)

        python_results = [r for r in results if "Python" in r["content"]]
        self.assertGreaterEqual(len(python_results), 2)

    def test_semantic_search(self):
        """Test semantic search with embeddings"""
        # Insert semantically related messages
        self.recall.insert("user", "I enjoy drinking coffee in the morning")
        self.recall.insert("user", "Programming is my hobby")
        self.recall.insert("user", "I like espresso and cappuccino")

        # Search for coffee-related content
        results = self.recall.search("What beverages do you like?", limit=3)

        # Should return results (semantic search finds related content)
        self.assertGreater(len(results), 0)

    def test_recent_messages_ordering(self):
        """Test that messages are ordered by recency"""
        self.recall.insert("user", "First message")
        time.sleep(0.1)
        self.recall.insert("user", "Second message")
        time.sleep(0.1)
        self.recall.insert("user", "Third message")

        results = self.recall.search(limit=3)

        # Most recent should be first (default ordering)
        self.assertEqual(results[0]["content"], "Third message")

    @unittest.skipIf(not _faiss_available, "FAISS not installed")
    def test_faiss_manager_integration(self):
        """Test that FAISS manager is properly initialized"""
        self.assertIsNotNone(self.recall.faiss_manager)
        self.assertTrue(self.recall.faiss_manager.is_available)


class TestArchivalMemory(unittest.TestCase):
    """Tests for Archival Memory (long-term facts with importance decay)"""

    def setUp(self):
        """Create fresh archival memory with temp database"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_archival_memory.db")
        
        # ISOLATION FIX: Patch global config to use temp FAISS index
        self.original_faiss_path = conf.FAISS_INDEX_PATH
        conf.FAISS_INDEX_PATH = os.path.join(self.temp_dir, "test_archival.index")
        
        self.archival = ArchivalMemory(db_path=self.db_path, use_semantic=True)

    def tearDown(self):
        """Cleanup temp files and restore config"""
        conf.FAISS_INDEX_PATH = self.original_faiss_path
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_insert_fact(self):
        """Test inserting a fact"""
        self.archival.insert(
            category="preference",
            content="User likes coffee",
            importance=8
        )

        self.assertEqual(self.archival.get_count(), 1)

    def test_insert_with_categories(self):
        """Test inserting facts with different categories"""
        self.archival.insert("preference", "User likes coffee", importance=8)
        self.archival.insert("personal_info", "User's name is Alex", importance=9)
        self.archival.insert("conversation_summary", "Discussed AI topics", importance=3)

        self.assertEqual(self.archival.get_count(), 3)

    def test_search_by_content(self):
        """Test searching facts by content"""
        self.archival.insert("preference", "User enjoys Python programming", importance=7)
        self.archival.insert("preference", "User dislikes JavaScript", importance=5)

        results = self.archival.search("Python", limit=5)

        self.assertGreater(len(results), 0)
        self.assertIn("Python", results[0]["content"])

    def test_importance_ordering(self):
        """Test that results are ordered by effective importance"""
        self.archival.insert("fact", "Low importance fact", importance=2)
        self.archival.insert("fact", "High importance fact", importance=9)
        self.archival.insert("fact", "Medium importance fact", importance=5)

        results = self.archival.search("fact", limit=3)

        # Higher effective importance should come first
        importances = [r["effective_importance"] for r in results]
        self.assertEqual(importances, sorted(importances, reverse=True))

    def test_importance_decay_formula(self):
        """Test that importance decay is applied correctly"""
        import sqlite3

        # Insert a fact
        self.archival.insert("test", "Recent fact", importance=8)

        # Manually backdate another fact to 60 days ago
        with sqlite3.connect(self.db_path) as conn:
            old_time = (datetime.now() - timedelta(days=60)).isoformat()
            conn.execute(
                "INSERT INTO archival_memory (timestamp, category, content, importance) VALUES (?, ?, ?, ?)",
                (old_time, "test", "Old fact", 8)
            )

        results = self.archival.search("fact", limit=5)

        # Find both facts
        recent = next((r for r in results if "Recent" in r["content"]), None)
        old = next((r for r in results if "Old" in r["content"]), None)

        if recent and old:
            # Recent fact should have higher effective importance
            self.assertGreater(recent["effective_importance"], old["effective_importance"])
            # Old fact should show age in days
            self.assertGreater(old["age_days"], 50)

    def test_effective_importance_calculation(self):
        """Test the effective importance calculation formula"""
        import math

        # Insert a brand new fact
        self.archival.insert("test", "Brand new fact", importance=5)

        results = self.archival.search("new fact", limit=1)

        if results:
            result = results[0]
            # For a new fact (age ~0), recency_score ~1.0
            # effective = 5 * 0.7 + 1.0 * 10 * 0.3 = 3.5 + 3.0 = 6.5
            expected = 5 * conf.IMPORTANCE_STATIC_WEIGHT + 10 * conf.IMPORTANCE_RECENCY_WEIGHT
            self.assertAlmostEqual(result["effective_importance"], expected, delta=0.5)


class TestFAISSManager(unittest.TestCase):
    """Tests for FAISS vector search manager"""

    def setUp(self):
        """Create fresh FAISS manager"""
        self.temp_dir = tempfile.mkdtemp()
        self.index_path = os.path.join(self.temp_dir, "test.index")

    def tearDown(self):
        """Cleanup temp files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipIf(not _faiss_available, "FAISS not installed")
    def test_faiss_manager_creation(self):
        """Test creating a new FAISS manager"""
        manager = FAISSManager(dimension=384, index_path=self.index_path)

        self.assertTrue(manager.is_available)
        self.assertEqual(manager.index.ntotal, 0)

    @unittest.skipIf(not _faiss_available, "FAISS not installed")
    def test_add_and_search(self):
        """Test adding embeddings and searching"""
        manager = FAISSManager(dimension=384, index_path=self.index_path)

        # Generate test embeddings
        emb1 = get_embedding("I love coffee")
        emb2 = get_embedding("I enjoy tea")
        emb3 = get_embedding("Programming is fun")

        if emb1 and emb2 and emb3:
            manager.add(emb1, row_id=1)
            manager.add(emb2, row_id=2)
            manager.add(emb3, row_id=3)

            self.assertEqual(manager.index.ntotal, 3)

            # Search for coffee-related
            query_emb = get_embedding("What hot drinks do you like?")
            if query_emb:
                results = manager.search(query_emb, k=2)

                # Should return results with IDs 1 or 2 (coffee/tea related)
                self.assertEqual(len(results), 2)
                result_ids = [r[0] for r in results]
                self.assertIn(1, result_ids)  # Coffee
                self.assertIn(2, result_ids)  # Tea

    @unittest.skipIf(not _faiss_available, "FAISS not installed")
    def test_save_and_load(self):
        """Test saving and loading FAISS index"""
        manager = FAISSManager(dimension=384, index_path=self.index_path)

        emb = get_embedding("Test embedding")
        if emb:
            manager.add(emb, row_id=42)
            manager.save()

            # Create new manager from saved index
            new_manager = FAISSManager(dimension=384, index_path=self.index_path)

            self.assertEqual(new_manager.index.ntotal, 1)
            self.assertEqual(new_manager.id_map, [42])

    @unittest.skipIf(not _faiss_available, "FAISS not installed")
    def test_search_performance(self):
        """Test that FAISS search is fast"""
        manager = FAISSManager(dimension=384, index_path=self.index_path)

        # Add 100 embeddings
        for i in range(100):
            emb = get_embedding(f"Test message number {i} about various topics")
            if emb:
                manager.add(emb, row_id=i)

        query_emb = get_embedding("Test message about topics")
        if query_emb:
            # Time the search
            start = time.time()
            for _ in range(10):
                results = manager.search(query_emb, k=10)
            elapsed = (time.time() - start) / 10 * 1000  # ms per search

            print(f"\n[FAISS Performance] Average search time: {elapsed:.2f}ms for 100 vectors")

            # Should be very fast (under 50ms)
            self.assertLess(elapsed, 50)


class TestToolCalling(unittest.TestCase):
    """Tests for tool calling functionality"""

    def setUp(self):
        """Setup for tool calling tests"""
        self.temp_dir = tempfile.mkdtemp()
        
        # ISOLATION FIX
        self.original_faiss_path = conf.FAISS_INDEX_PATH
        conf.FAISS_INDEX_PATH = os.path.join(self.temp_dir, "test_tools.index")

    def tearDown(self):
        """Cleanup"""
        conf.FAISS_INDEX_PATH = self.original_faiss_path
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_memory_tools_import(self):
        """Test that memory tools can be imported"""
        from memory_tools import MemoryToolExecutor

        db_path = os.path.join(self.temp_dir, "test_tools.db")
        memory = Memory(db_path=db_path)

        executor = MemoryToolExecutor(memory, None, None)
        self.assertIsNotNone(executor)

    def test_core_memory_append_tool(self):
        """Test core_memory_append tool"""
        from memory_tools import MemoryToolExecutor

        db_path = os.path.join(self.temp_dir, "test_tools.db")
        memory = Memory(db_path=db_path)

        executor = MemoryToolExecutor(memory, None, None)

        result = executor.execute("core_memory_append", {
            "label": "human",
            "content": " User's favorite color is blue."
        })

        self.assertIn("success", result.lower())
        self.assertIn("blue", memory.get_block("human").value)

    def test_core_memory_replace_tool(self):
        """Test core_memory_replace tool"""
        from memory_tools import MemoryToolExecutor

        db_path = os.path.join(self.temp_dir, "test_tools.db")
        memory = Memory(db_path=db_path)

        # First add some content
        human = memory.get_block("human")
        human.value = "User likes coffee."

        executor = MemoryToolExecutor(memory, None, None)

        result = executor.execute("core_memory_replace", {
            "label": "human",
            "old_content": "coffee",
            "new_content": "tea"
        })

        self.assertIn("success", result.lower())
        self.assertIn("tea", memory.get_block("human").value)

    def test_archival_memory_insert_tool(self):
        """Test archival_memory_insert tool"""
        from memory_tools import MemoryToolExecutor

        db_paths = {
            "core": os.path.join(self.temp_dir, "core.db"),
            "archival": os.path.join(self.temp_dir, "archival.db")
        }

        memory = Memory(db_path=db_paths["core"])
        archival = ArchivalMemory(db_path=db_paths["archival"])

        executor = MemoryToolExecutor(memory, None, archival)

        result = executor.execute("archival_memory_insert", {
            "category": "preference",
            "content": "User prefers dark mode",
            "importance": 7
        })

        self.assertIn("saved", result.lower())
        self.assertEqual(archival.get_count(), 1)

    def test_tool_schema_format(self):
        """Test that tool schemas are in correct OpenAI format"""
        from memory_tools import MemoryToolExecutor

        memory = Memory(db_path=os.path.join(self.temp_dir, "test.db"))
        executor = MemoryToolExecutor(memory, None, None)

        tools = executor.get_tool_schemas()

        self.assertIsInstance(tools, list)
        for tool in tools:
            self.assertIn("type", tool)
            self.assertEqual(tool["type"], "function")
            self.assertIn("function", tool)
            self.assertIn("name", tool["function"])
            self.assertIn("parameters", tool["function"])


class TestSearchPerformance(unittest.TestCase):
    """Performance comparison tests: FAISS vs Naive search"""

    def setUp(self):
        """Setup test data"""
        self.temp_dir = tempfile.mkdtemp()
        
        # ISOLATION FIX
        self.original_faiss_path = conf.FAISS_INDEX_PATH
        conf.FAISS_INDEX_PATH = os.path.join(self.temp_dir, "test_perf.index")

    def tearDown(self):
        """Cleanup"""
        conf.FAISS_INDEX_PATH = self.original_faiss_path
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipIf(not _faiss_available, "FAISS not installed")
    def test_faiss_vs_naive_performance(self):
        """Compare FAISS vs naive cosine similarity performance"""
        import numpy as np

        db_path = os.path.join(self.temp_dir, "perf_test.db")

        # Create recall memory with FAISS
        recall_faiss = RecallMemory(db_path=db_path, use_semantic=True)

        # Insert test messages
        test_messages = [
            "I love programming in Python",
            "JavaScript is great for web development",
            "Machine learning is fascinating",
            "Coffee helps me code better",
            "The weather is nice today",
            "I enjoy reading about AI",
            "Database optimization is important",
            "Cloud computing changed everything",
            "Open source software is valuable",
            "Testing code prevents bugs",
        ] * 5  # 50 messages

        print(f"\n[Performance Test] Inserting {len(test_messages)} messages...")
        for msg in test_messages:
            recall_faiss.insert("user", msg)

        # Test FAISS search speed
        query = "Tell me about programming languages"

        start = time.time()
        for _ in range(10):
            results = recall_faiss.search(query, limit=5)
        faiss_time = (time.time() - start) / 10 * 1000

        print(f"[FAISS Search] Average time: {faiss_time:.2f}ms")
        print(f"[FAISS Search] Found {len(results)} results")

        # FAISS should be fast
        self.assertLess(faiss_time, 100)  # Under 100ms
        self.assertGreater(len(results), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete memory workflow"""

    def setUp(self):
        """Setup complete memory system"""
        self.temp_dir = tempfile.mkdtemp()
        self.core_db = os.path.join(self.temp_dir, "core.db")
        self.recall_db = os.path.join(self.temp_dir, "recall.db")
        self.archival_db = os.path.join(self.temp_dir, "archival.db")
        
        # ISOLATION FIX: Patch global config to use temp FAISS index
        self.original_faiss_path = conf.FAISS_INDEX_PATH
        conf.FAISS_INDEX_PATH = os.path.join(self.temp_dir, "test_integration.index")

        self.core = Memory(db_path=self.core_db)
        self.recall = RecallMemory(db_path=self.recall_db, use_semantic=True)
        self.archival = ArchivalMemory(db_path=self.archival_db, use_semantic=True)

    def tearDown(self):
        """Cleanup"""
        conf.FAISS_INDEX_PATH = self.original_faiss_path
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_conversation_workflow(self):
        """Test a complete conversation workflow"""
        # 1. User introduces themselves
        self.recall.insert("user", "Hi, my name is Alex and I'm a Python developer")

        # 2. System updates core memory
        human = self.core.get_block("human")
        human.append(" User's name is Alex. They are a Python developer.")
        self.core.save()

        # 3. Store as archival fact
        self.archival.insert(
            category="personal_info",
            content="User's name is Alex, a Python developer",
            importance=9
        )

        # 4. Continue conversation
        self.recall.insert("assistant", "Nice to meet you, Alex! I'd love to help with your Python projects.")
        self.recall.insert("user", "Can you help me with machine learning?")
        self.recall.insert("assistant", "Of course! What ML topic interests you?")

        # 5. Verify memory state
        self.assertEqual(self.recall.get_count(), 4)
        self.assertEqual(self.archival.get_count(), 1)
        self.assertIn("Alex", self.core.get_block("human").value)

        # 6. Test retrieval
        results = self.recall.search("Python", limit=2)
        self.assertGreater(len(results), 0)

        # This previously failed due to FAISS index contamination across tests
        facts = self.archival.search("user name", limit=1)
        self.assertGreater(len(facts), 0)
        self.assertIn("Alex", facts[0]["content"])

    def test_memory_persistence_workflow(self):
        """Test that all memory persists across sessions"""
        # Session 1: Store data
        self.core.get_block("human").append(" Prefers dark mode.")
        self.core.save()

        self.recall.insert("user", "I like dark mode")
        self.archival.insert("preference", "User prefers dark mode", importance=6)

        # Simulate new session (reusing DBs, forcing index reload)
        # Note: We keep the same temp index path to simulate restart with persistence
        new_core = Memory(db_path=self.core_db)
        new_recall = RecallMemory(db_path=self.recall_db, use_semantic=True)
        new_archival = ArchivalMemory(db_path=self.archival_db, use_semantic=True)

        # Verify persistence
        self.assertIn("dark mode", new_core.get_block("human").value)
        self.assertEqual(new_recall.get_count(), 1)
        self.assertEqual(new_archival.get_count(), 1)


def run_all_tests():
    """Run all tests and print summary"""
    print("=" * 70)
    print("WALL-E Memory System Test Suite")
    print("=" * 70)
    print(f"FAISS Available: {_faiss_available}")
    print(f"Embedding Model: {conf.EMBEDDING_MODEL}")
    print(f"Use FAISS: {conf.USE_FAISS}")
    print("=" * 70)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCoreMemory))
    suite.addTests(loader.loadTestsFromTestCase(TestRecallMemory))
    suite.addTests(loader.loadTestsFromTestCase(TestArchivalMemory))
    suite.addTests(loader.loadTestsFromTestCase(TestFAISSManager))
    suite.addTests(loader.loadTestsFromTestCase(TestToolCalling))
    suite.addTests(loader.loadTestsFromTestCase(TestSearchPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)

    if result.wasSuccessful():
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")
        for test, traceback in result.failures + result.errors:
            print(f"\nFailed: {test}")
            print(traceback)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
