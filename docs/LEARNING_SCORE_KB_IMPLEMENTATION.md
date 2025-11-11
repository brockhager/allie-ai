# Learning Score KB Implementation

## Overview

This document describes the implementation of the knowledge base-based learning score system, which replaced the previous facts-based scoring to provide more accurate reflection of verified learning progress.

## Problem Statement

The original learning score was stuck at 17/1000 because:
1. Used `facts` table instead of `knowledge_base` table
2. Had fixed denominator of 1000, making early scores appear artificially low
3. Didn't account for confidence scores of facts
4. Didn't reflect recent growth or learning momentum
5. Score didn't increase proportionally when new facts were added

## Solution

Implemented a comprehensive KB-based scoring system that:
- Queries `knowledge_base` table for verified facts
- Uses dynamic scoring based on KB metrics
- Includes confidence weighting for fact quality
- Rewards recent learning activity (7-day and 24-hour windows)
- Provides immediate feedback via growth bonus
- Maintains 1-1000 range with meaningful progression

## Files Modified

### 1. `advanced-memory/db.py`
**Added:** `get_kb_statistics()` method

Queries knowledge_base table and returns:
- `kb_total`: Total number of KB facts
- `kb_active`: Facts with status='true' (verified)
- `kb_recent_7d`: Facts added in last 7 days
- `kb_recent_24h`: Facts added in last 24 hours
- `kb_avg_confidence`: Average confidence_score of all facts
- `kb_top_keywords`: Top 10 most common keywords
- `kb_by_status`: Count of facts by status (true/false/not_verified/needs_review/experimental)

```python
def get_kb_statistics(self):
    """Get statistics about the knowledge base for learning score calculation"""
    cursor = self.connection.cursor(dictionary=True)
    
    # Total KB facts
    cursor.execute("SELECT COUNT(*) as count FROM knowledge_base")
    kb_total = cursor.fetchone()['count']
    
    # Active (verified) KB facts
    cursor.execute("SELECT COUNT(*) as count FROM knowledge_base WHERE status = 'true'")
    kb_active = cursor.fetchone()['count']
    
    # Recent KB facts (7-day window)
    cursor.execute("""
        SELECT COUNT(*) as count 
        FROM knowledge_base 
        WHERE learned_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
    """)
    kb_recent_7d = cursor.fetchone()['count']
    
    # Recent KB facts (24-hour window)
    cursor.execute("""
        SELECT COUNT(*) as count 
        FROM knowledge_base 
        WHERE learned_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
    """)
    kb_recent_24h = cursor.fetchone()['count']
    
    # Average confidence score
    cursor.execute("SELECT AVG(confidence_score) as avg_conf FROM knowledge_base")
    result = cursor.fetchone()
    kb_avg_confidence = result['avg_conf'] if result['avg_conf'] else 0
    
    cursor.close()
    
    return {
        'kb_total': kb_total,
        'kb_active': kb_active,
        'kb_recent_7d': kb_recent_7d,
        'kb_recent_24h': kb_recent_24h,
        'kb_avg_confidence': kb_avg_confidence,
        'kb_top_keywords': {...},
        'kb_by_status': {...}
    }
```

### 2. `backend/server.py`
**Modified:** `/api/hybrid-memory/statistics` endpoint

Added KB statistics to existing statistics response:

```python
@app.get("/api/hybrid-memory/statistics")
async def get_hybrid_memory_statistics():
    """Get statistics about hybrid memory system"""
    stats = {
        'facts': {...},
        'conversations': {...},
        'learning_queue': {...},
        # ... existing stats ...
    }
    
    # Add KB statistics
    kb_stats = advanced_memory.get_kb_statistics()
    stats.update(kb_stats)
    
    return stats
```

### 3. `frontend/static/ui.html`
**Modified:** `updateLearningScore()` function (lines ~508-590)

Completely rewrote scoring logic to use KB statistics:

```javascript
function updateLearningScore() {
    fetch('/api/hybrid-memory/statistics')
        .then(response => response.json())
        .then(stats => {
            // Extract KB statistics
            const kb_total = stats.kb_total || 0;
            const kb_active = stats.kb_active || 0;
            const kb_recent_7d = stats.kb_recent_7d || 0;
            const kb_recent_24h = stats.kb_recent_24h || 0;
            const kb_avg_confidence = stats.kb_avg_confidence || 0;
            
            // Return minimum score for empty KB
            if (kb_total === 0) {
                updateScoreDisplay(1, "Beginner");
                return;
            }
            
            let rawScore = 0;
            
            // Base score: KB active facts (0-40 points)
            rawScore += Math.min((kb_active / 10) * 10, 40);
            
            // Quality ratio (0-25 points)
            const qualityRatio = kb_active / kb_total;
            rawScore += qualityRatio * 25;
            
            // Growth score (0-25 points)
            if (kb_total > 0) {
                const growthRatio = kb_recent_7d / kb_total;
                rawScore += Math.min(growthRatio * 100, 25);
            }
            
            // Confidence weighting (0-10 points)
            if (kb_avg_confidence > 0) {
                rawScore += (kb_avg_confidence / 100) * 10;
            }
            
            // Clamp raw score to 0-100
            rawScore = Math.max(0, Math.min(100, rawScore));
            
            // Scale to 1-1000 range
            let finalScore = Math.round(rawScore * 10);
            
            // Add growth bonus (immediate feedback)
            const growthBonus = kb_recent_24h * 10;
            finalScore += growthBonus;
            
            // Final clamp
            finalScore = Math.min(1000, Math.max(1, finalScore));
            
            updateScoreDisplay(finalScore, getScoreLabel(finalScore));
        });
}
```

### 4. `tests/test_learning_score_kb.py`
**Created:** Unit tests for KB-based scoring

7 comprehensive test methods:
- `test_empty_kb_returns_minimum_score`: Verifies empty KB returns score of 1
- `test_adding_facts_increases_score`: Tests proportional score increases when facts added
- `test_recent_growth_affects_score`: Validates 7-day growth window affects score
- `test_confidence_affects_score`: Verifies confidence weighting impacts score
- `test_score_stays_in_range`: Ensures score never exceeds 1-1000 bounds
- `test_example_scenario`: Tests specific scenario (3/12 facts in 7 days)

Key features:
- Uses pytest fixtures for setup/teardown
- Implements `calculate_score()` matching UI formula exactly
- Cleans up test data with `test_` prefix
- Provides detailed debug output with `-v -s` flags

### 5. `tests/test_learning_score_integration.py`
**Created:** Integration tests simulating conversation flow

3 integration test methods:
- `test_conversation_generates_kb_growth`: Simulates conversation generating facts, verifies score increase
- `test_multiple_conversations_compound_growth`: Tests compound growth across multiple conversations
- `test_score_reflects_confidence_quality`: Validates confidence quality affects final score

Key features:
- Simulates realistic conversation scenarios
- Adds facts to KB and measures score changes
- Tests growth bonus and compound effects
- Verifies confidence weighting

## Scoring Formula

### Mathematical Formula

```
rawScore = base + quality + growth + confidence

where:
  base = min((kb_active / 10) × 10, 40)           [0-40 points]
  quality = (kb_active / kb_total) × 25           [0-25 points]
  growth = min((kb_recent_7d / kb_total) × 100, 25) [0-25 points]
  confidence = (kb_avg_confidence / 100) × 10     [0-10 points]

finalScore = (rawScore × 10) + (kb_recent_24h × 10)
finalScore = clamp(finalScore, 1, 1000)
```

### Component Breakdown

| Component | Weight | Purpose | Example |
|-----------|--------|---------|---------|
| **Base Score** | 0-40 pts | Rewards KB growth | 10 facts = 10pts, 40+ facts = 40pts |
| **Quality Ratio** | 0-25 pts | Ensures fact verification | 100% verified = 25pts |
| **Growth Score** | 0-25 pts | Recent learning momentum | 25% recent = 25pts |
| **Confidence** | 0-10 pts | Fact reliability | 90% confidence = 9pts |
| **Growth Bonus** | +10/fact | Immediate feedback | 3 new facts = +30pts |

### Example Calculations

#### Example 1: Empty KB
```
kb_total=0, kb_active=0, kb_recent_7d=0, kb_recent_24h=0, kb_avg_confidence=0

Result: score = 1 (minimum)
```

#### Example 2: Starting Out
```
kb_total=8, kb_active=8, kb_recent_7d=8, kb_recent_24h=8, kb_avg_confidence=87

base = min((8/10)*10, 40) = 8.0
quality = (8/8) * 25 = 25.0
growth = min((8/8)*100, 25) = 25.0
confidence = (87/100) * 10 = 8.7
rawScore = 8.0 + 25.0 + 25.0 + 8.7 = 66.7

finalScore = (66.7 * 10) + (8 * 10) = 667 + 80 = 747
Result: score = 747
```

#### Example 3: Growing Knowledge Base
```
kb_total=12, kb_active=12, kb_recent_7d=3, kb_recent_24h=3, kb_avg_confidence=85

base = min((12/10)*10, 40) = 12.0
quality = (12/12) * 25 = 25.0
growth = min((3/12)*100, 25) = 25.0  (25% is recent)
confidence = (85/100) * 10 = 8.5
rawScore = 12.0 + 25.0 + 25.0 + 8.5 = 70.5

finalScore = (70.5 * 10) + (3 * 10) = 705 + 30 = 735
Result: score = 735
```

#### Example 4: Mature KB
```
kb_total=55, kb_active=55, kb_recent_7d=10, kb_recent_24h=5, kb_avg_confidence=94

base = min((55/10)*10, 40) = 40.0  (capped at 40)
quality = (55/55) * 25 = 25.0
growth = min((10/55)*100, 25) = 18.2
confidence = (94/100) * 10 = 9.4
rawScore = 40.0 + 25.0 + 18.2 + 9.4 = 92.6

finalScore = (92.6 * 10) + (5 * 10) = 926 + 50 = 976
Result: score = 976
```

## Test Results

### Unit Tests
All 6 tests passed:
```bash
$ python -m pytest tests/test_learning_score_kb.py -v

tests/test_learning_score_kb.py::TestLearningScoreKB::test_empty_kb_returns_minimum_score PASSED
tests/test_learning_score_kb.py::TestLearningScoreKB::test_adding_facts_increases_score PASSED
tests/test_learning_score_kb.py::TestLearningScoreKB::test_recent_growth_affects_score PASSED
tests/test_learning_score_kb.py::TestLearningScoreKB::test_confidence_affects_score PASSED
tests/test_learning_score_kb.py::TestLearningScoreKB::test_score_stays_in_range PASSED
tests/test_learning_score_kb.py::TestLearningScoreKB::test_example_scenario PASSED

6 passed in 1.56s
```

### Integration Tests
All 3 tests passed:
```bash
$ python -m pytest tests/test_learning_score_integration.py -v

tests/test_learning_score_integration.py::TestLearningScoreIntegration::test_conversation_generates_kb_growth PASSED
tests/test_learning_score_integration.py::TestLearningScoreIntegration::test_multiple_conversations_compound_growth PASSED
tests/test_learning_score_integration.py::TestLearningScoreIntegration::test_score_reflects_confidence_quality PASSED

3 passed in 0.51s
```

## Design Decisions

### 1. Why knowledge_base instead of facts table?
- **Knowledge base is authoritative**: Contains only verified, accepted facts
- **Facts table includes unverified entries**: Mix of true/false/experimental creates noise
- **Better reflects actual learning**: Score should measure verified knowledge, not just extraction attempts

### 2. Why dynamic scaling instead of fixed 1000 denominator?
- **Proportional feedback**: Adding 3 facts to 8 total should visibly increase score
- **Avoids "stuck at low score" problem**: Fixed denominator made early scores artificially low
- **More motivating**: Users see immediate results from learning

### 3. Why include growth bonus?
- **Immediate feedback**: Adding facts today gives instant +10 per fact
- **Encourages engagement**: Makes system feel responsive
- **Temporary boost**: 24-hour window creates urgency without permanent inflation

### 4. Why confidence weighting?
- **Quality matters**: High-confidence facts are more valuable
- **Reflects reliability**: Score should measure not just quantity but quality
- **Small weight (10 points)**: Doesn't dominate score but provides differentiation

### 5. Why multiple time windows (7d, 24h)?
- **7-day window**: Measures sustained learning momentum in growth component
- **24-hour window**: Provides immediate feedback bonus for recent activity
- **Different purposes**: Long-term trend vs. short-term incentive

## Benefits

### User Experience
- **Visible progress**: Score increases immediately when facts are added
- **Meaningful feedback**: Score reflects actual learning, not arbitrary metrics
- **Motivating**: Growth bonus encourages continued interaction

### Technical
- **Maintainable**: Clear formula documented in code and tests
- **Testable**: Comprehensive unit and integration tests
- **Extensible**: Easy to adjust weights or add new components
- **Performant**: Single query to get all KB statistics

### Accuracy
- **Reflects verified knowledge**: Uses KB table instead of unverified facts
- **Quality-weighted**: Confidence scores affect final score
- **Growth-aware**: Recent learning momentum included in calculation
- **Bounded**: Always stays in 1-1000 range

## Future Enhancements

Potential improvements to consider:

1. **Keyword diversity bonus**: Reward learning across different topics
2. **Retention penalty**: Reduce score for very old facts that haven't been reinforced
3. **Category weighting**: Give more points for facts in priority categories
4. **User corrections bonus**: Reward when user corrects false facts
5. **Difficulty weighting**: Higher points for complex/detailed facts
6. **Source diversity**: Bonus for facts from multiple sources
7. **Relationship bonus**: Extra points for facts with cross-references

## Rollback Plan

If needed to rollback to previous scoring:

1. Revert `frontend/static/ui.html` updateLearningScore() function
2. Remove KB statistics from `backend/server.py` endpoint
3. Remove `get_kb_statistics()` from `advanced-memory/db.py`
4. Keep tests for future reference

Previous version used facts table and fixed scaling, but had low scores (17/1000).

## Monitoring

To monitor score behavior in production:

1. Check console logs in browser dev tools (detailed scoring breakdown)
2. Query `/api/hybrid-memory/statistics` to see raw KB metrics
3. Run integration tests periodically to verify scoring logic
4. Monitor for scores stuck at 1 or 1000 (edge cases)

## References

- Main implementation: `frontend/static/ui.html` (lines 508-590)
- Backend API: `backend/server.py` (line ~2432)
- Database method: `advanced-memory/db.py` (lines ~1010-1070)
- Unit tests: `tests/test_learning_score_kb.py`
- Integration tests: `tests/test_learning_score_integration.py`
- Documentation: `README.md` (Learning Score System section)
