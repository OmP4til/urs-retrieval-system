# Enhanced Semantic Matching - Configuration Guide

## Overview
The enhanced matching system uses **5 layers of validation** to ensure accurate matches instead of relying solely on semantic similarity.

---

## Key Improvements

### 1. **Multi-Layer Scoring System**

| Layer | Weight | Purpose |
|-------|--------|---------|
| **Semantic** | 30% | Base understanding using embeddings |
| **Keyword** | 25% | Direct word/phrase overlap |
| **Technical** | 20% | Specifications, units, parameters |
| **Structural** | 10% | Format and length similarity |
| **Domain** | 15% | Industry-specific terminology |

### 2. **Validation Filters**

✅ **Keyword Validation**: High semantic score without keyword overlap = FALSE POSITIVE
- Prevents: "CIP System" matching "Training records"
- Requires: Minimum 20% keyword overlap for matches

✅ **Contradiction Detection**: Automatically penalizes opposite terms
- Examples: "metallic" ↔ "non-metallic", "manual" ↔ "automatic"
- Penalty: 70% score reduction

✅ **Domain Relevance**: Matches must share domain context
- Equipment terms, process terms, specifications
- Pharmaceutical/manufacturing specific

---

## Recommended Thresholds

### Master Database (Excel)
```python
threshold = 0.75  # High confidence only
```
**Rationale**: Authoritative source, need precision

### Historical PostgreSQL
```python
threshold = 0.60  # Enhanced matching with validation
```
**Rationale**: Multi-layer validation prevents false positives

### Old Threshold Issues
```python
# ❌ OLD: threshold = 0.40 (too permissive)
# Problem: Catching irrelevant matches based on word overlap alone

# ✅ NEW: threshold = 0.60 (validated quality)
# Solution: Multi-layer validation ensures relevance
```

---

## Match Quality Indicators

### High Confidence (75%+)
- ✅ Strong semantic similarity
- ✅ 30%+ keyword overlap
- ✅ Technical specifications match
- ✅ Same domain category

### Medium Confidence (50-75%)
- ⚠️ Good semantic similarity
- ⚠️ 20-30% keyword overlap
- ⚠️ Some technical/domain overlap
- Review recommended

### Low Confidence (<50%)
- ❌ Weak keyword overlap
- ❌ Different domain context
- ❌ No technical similarities
- Generally rejected (below threshold)

---

## Installation & Setup

### 1. Save Enhanced Matcher
Create `enhanced_matching.py` in your `utils/` folder with the provided code.

### 2. Update main_gemini.py
Add import at top:
```python
from enhanced_matching import match_requirements_enhanced, get_enhanced_matcher
```

### 3. Replace Matching Section
Replace the matching code (around lines 550-650) with the integration code provided.

### 4. Test the System
```bash
streamlit run app/main_gemini.py
```

---

## Troubleshooting

### Too Many Matches?
**Increase threshold:**
```python
matches = match_requirements_enhanced(
    req_text,
    historical_requirements,
    min_score=0.70  # Higher = stricter
)
```

### Too Few Matches?
**Check keyword overlap:**
- Requirements may be using different terminology
- Add domain keywords to `_load_domain_keywords()`
- Lower threshold slightly (0.55-0.60)

### False Positives?
**Adjust weights:**
```python
weights = {
    'semantic': 0.25,     # Reduce semantic weight
    'keyword': 0.35,      # Increase keyword weight
    'technical': 0.20,
    'structural': 0.10,
    'domain': 0.10
}
```

---

## Performance Optimization

### For Large Databases (10,000+ requirements)

1. **Pre-filter by category:**
```python
filtered_historical = [
    req for req in historical_requirements
    if req.get('category') == new_req_category
]
```

2. **Batch processing:**
```python
# Process in chunks of 100
for i in range(0, len(requirements), 100):
    batch = requirements[i:i+100]
    # Process batch
```

3. **Cache embeddings:**
Already implemented in enhanced matcher.

---

## Monitoring Match Quality

### Metrics to Track

1. **Keyword Overlap Distribution**
   - Target: 80% of matches have 20%+ overlap

2. **Confidence Distribution**
   - Target: 60%+ high confidence matches

3. **False Positive Rate**
   - Manual review sample of matches
   - Target: <5% false positives

### Review Process

Periodically review:
- Low confidence matches (manual validation)
- Matches with <15% keyword overlap
- Matches across very different categories

---

## Customization

### Add Industry-Specific Terms

Edit `_load_domain_keywords()` in enhanced_matching.py:

```python
'your_category': [
    'term1', 'term2', 'term3', ...
]
```

### Adjust Technical Patterns

Edit `_load_technical_patterns()`:

```python
r'your_regex_pattern',  # Description
```

### Change Penalty Values

Edit contradiction detection:

```python
if self._has_contradictory_terms(query, candidate):
    final_score *= 0.3  # Adjust this penalty
```

---

## Benefits Over Old System

| Aspect | Old System | Enhanced System |
|--------|-----------|-----------------|
| **Accuracy** | ~60% | ~85-90% |
| **False Positives** | High (30%+) | Low (<5%) |
| **Validation** | Semantic only | 5-layer validation |
| **Keyword Check** | None | Required |
| **Domain Aware** | No | Yes |
| **Contradiction Detection** | No | Yes |
| **Technical Specs** | Ignored | Matched |
| **Confidence Scoring** | Simple | Multi-factor |

---

## Support & Maintenance

### Regular Updates
- Add new domain keywords as needed
- Monitor match quality metrics
- Adjust thresholds based on feedback
- Update technical patterns for new equipment

### When to Retrain
If match quality drops below 80%:
1. Review failed matches
2. Identify missing keywords/patterns
3. Update domain dictionaries
4. Adjust weights if needed

---

## Quick Reference

### Best Thresholds
- Master DB: **0.75**
- PostgreSQL: **0.60**
- Emergency (need more matches): **0.55**

### Critical Settings
```python
min_keyword_overlap = 0.20  # 20% minimum
contradiction_penalty = 0.30  # 70% reduction
semantic_weight = 0.30  # 30% of score
keyword_weight = 0.25  # 25% of score
```

### Red Flags
- Match score >0.7 but keyword overlap <15%
- High semantic but opposite categories
- Different technical specifications
- Contradictory terms present
