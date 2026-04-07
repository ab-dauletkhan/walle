"""
Memory Inspector - See what's ACTUALLY stored in WALL-E's memory
Use this to diagnose memory issues
"""

import sqlite3
import json
from pathlib import Path


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def inspect_core_memory():
    """Inspect core memory database"""
    print_section("📦 CORE MEMORY (Always Visible)")
    
    db_path = "walle_core_memory.db"
    if not Path(db_path).exists():
        print("❌ Core memory database not found!")
        print("   Run WALL-E at least once to create it.")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT label, value, limit_chars, read_only FROM core_memory")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("⚠️  Core memory is empty!")
        return
    
    for label, value, limit_chars, read_only in rows:
        readonly_marker = " [READ-ONLY]" if read_only else ""
        print(f"\n📝 Block: {label}{readonly_marker}")
        print(f"   Size: {len(value)}/{limit_chars} chars")
        print(f"   Content:")
        print(f"   {'-' * 66}")
        # Print value with indentation
        for line in value.split('\n'):
            print(f"   {line}")
        print(f"   {'-' * 66}")


def inspect_recall_memory():
    """Inspect recall memory database"""
    print_section("💭 RECALL MEMORY (Recent Conversations)")
    
    db_path = "walle_recall_memory.db"
    if not Path(db_path).exists():
        print("❌ Recall memory database not found!")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get count
    cursor.execute("SELECT COUNT(*) FROM recall_memory")
    count = cursor.fetchone()[0]
    print(f"\n📊 Total conversations: {count}")
    
    if count == 0:
        print("⚠️  No conversations stored yet!")
        conn.close()
        return
    
    # Get recent entries
    cursor.execute("""
        SELECT id, timestamp, role, content, tools_used
        FROM recall_memory
        ORDER BY timestamp DESC
        LIMIT 10
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    print(f"\n📜 Last {len(rows)} entries:")
    for i, (id, timestamp, role, content, tools_used) in enumerate(rows, 1):
        tools = json.loads(tools_used) if tools_used else []
        tools_str = f" [Tools: {', '.join(tools)}]" if tools else ""
        
        print(f"\n{i}. [{role.upper()}] {timestamp}{tools_str}")
        # Truncate long content
        if len(content) > 200:
            print(f"   {content[:200]}...")
        else:
            print(f"   {content}")


def inspect_archival_memory():
    """Inspect archival memory database"""
    print_section("📚 ARCHIVAL MEMORY (Long-term Storage)")
    
    db_path = "walle_archival_memory.db"
    if not Path(db_path).exists():
        print("❌ Archival memory database not found!")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get count
    cursor.execute("SELECT COUNT(*) FROM archival_memory")
    count = cursor.fetchone()[0]
    print(f"\n📊 Total entries: {count}")
    
    if count == 0:
        print("⚠️  No archival memories stored yet!")
        conn.close()
        return
    
    # Get entries by category
    cursor.execute("""
        SELECT category, COUNT(*) as count
        FROM archival_memory
        GROUP BY category
        ORDER BY count DESC
    """)
    
    categories = cursor.fetchall()
    print(f"\n📂 Categories:")
    for category, cat_count in categories:
        print(f"   - {category}: {cat_count} entries")
    
    # Get all entries
    cursor.execute("""
        SELECT id, timestamp, category, content, importance
        FROM archival_memory
        ORDER BY importance DESC, timestamp DESC
        LIMIT 20
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    if rows:
        print(f"\n📝 Stored memories (top {len(rows)} by importance):")
        for i, (id, timestamp, category, content, importance) in enumerate(rows, 1):
            importance_bar = "★" * importance + "☆" * (10 - importance)
            print(f"\n{i}. [{category}] {importance_bar} {importance}/10")
            print(f"   {timestamp}")
            if len(content) > 150:
                print(f"   {content[:150]}...")
            else:
                print(f"   {content}")


def analyze_memory_issues():
    """Analyze potential memory issues"""
    print_section("🔍 DIAGNOSTIC ANALYSIS")
    
    # Check core memory for human block
    db_path = "walle_core_memory.db"
    if Path(db_path).exists():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM core_memory WHERE label='human'")
        result = cursor.fetchone()
        conn.close()
        
        if result:
            human_block = result[0]
            print("\n🔍 Checking 'human' block in core memory:")
            
            # Check if it's default/empty
            if "operator who gives me movement commands" in human_block:
                print("   ⚠️  WARNING: Human block is still default/empty!")
                print("   This means WALL-E has NOT stored any info about you.")
                print("   ❌ BUG CONFIRMED: Memory tools are not being called!")
            elif len(human_block) < 100:
                print("   ⚠️  Human block is very short:")
                print(f"   '{human_block}'")
                print("   May need more information stored.")
            else:
                print("   ✅ Human block contains information:")
                print(f"   {human_block[:200]}...")
        else:
            print("   ❌ Human block not found!")
    
    # Check if recall memory has tool calls
    db_path = "walle_recall_memory.db"
    if Path(db_path).exists():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT tools_used FROM recall_memory WHERE tools_used IS NOT NULL")
        tool_calls = cursor.fetchall()
        conn.close()
        
        print("\n🔍 Checking tool usage in recall memory:")
        if tool_calls:
            all_tools = []
            for (tools_json,) in tool_calls:
                all_tools.extend(json.loads(tools_json))
            
            memory_tools = [t for t in all_tools if 'memory' in t]
            
            print(f"   📊 Total tool calls: {len(all_tools)}")
            print(f"   💾 Memory tool calls: {len(memory_tools)}")
            
            if memory_tools:
                print(f"   ✅ Memory tools ARE being used:")
                for tool in set(memory_tools):
                    print(f"      - {tool}")
            else:
                print(f"   ⚠️  WARNING: No memory tools detected!")
                print(f"   Available tools used: {set(all_tools)}")
        else:
            print("   ℹ️  No tool calls recorded yet")


def main():
    print("=" * 70)
    print("  🔬 WALL-E Memory Inspector")
    print("  Diagnostic tool to see what's actually stored")
    print("=" * 70)
    
    # Inspect all memory tiers
    inspect_core_memory()
    inspect_recall_memory()
    inspect_archival_memory()
    
    # Analyze issues
    analyze_memory_issues()
    
    # Summary
    print_section("📋 SUMMARY")
    print("""
Next steps:
1. Check the 'human' block in Core Memory above
2. If it's default/empty → Memory tools NOT being called ❌
3. If it has your info → Storage works, check retrieval ✅
4. Check recall memory for tool usage patterns
5. Run QUICK_TESTS.md to identify specific failing commands
    """)
    
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
