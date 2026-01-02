# Editable Deviations Column - Feature Guide

## 🎯 New Feature: Interactive Deviations Column

### What's New

Added an **editable "Deviations / Response" column** to the requirement matching table that allows you to:

1. ✅ **View Master DB responses** automatically populated for matched requirements
2. ✏️ **Edit or add your own responses** for requirements without matches
3. 📥 **Export the complete table** with your additions as CSV

### How It Works

#### Column Behavior

| Scenario | What Appears in "Deviations" Column | What You Can Do |
|----------|-------------------------------------|-----------------|
| Master DB Match Found | Auto-populated with Master DB response | View the response (read-only for matched items) |
| PostgreSQL Match Found | Empty (no Master DB response) | Add your own response |
| No Match Found | Empty | Add your own response |

#### Visual Layout

```
📊 Multi-Layer Requirement Matching Results
┌─────────────────────────────────────────────────────────────────────────────┐
│ New Requirement │ Category │ Priority │ Deviations/Response │ Has Match │... │
├─────────────────────────────────────────────────────────────────────────────┤
│ CIP System      │ Cleaning │ High     │ WIP System will be  │ Yes -     │... │
│                 │          │          │ provided            │ Master DB │    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Training records│ Data     │ Medium   │ [EMPTY - CLICK TO   │ Yes -     │... │
│ maintained      │          │          │  EDIT AND ADD YOUR  │PostgreSQL │    │
│                 │          │          │  RESPONSE]          │           │    │
├─────────────────────────────────────────────────────────────────────────────┤
│ New unique req  │ Process  │ Low      │ [EMPTY - CLICK TO   │ No        │... │
│                 │          │          │  EDIT AND ADD YOUR  │           │    │
│                 │          │          │  RESPONSE]          │           │    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Column Configuration

- **Width:** Large (to accommodate detailed responses)
- **Max Characters:** 500
- **Editable:** Only the "Deviations" column
- **All other columns:** Read-only (locked)

### Usage Instructions

#### Step 1: Process Your Document
1. Select "🔍 Extract + Match Requirements" mode
2. Upload your URS DOCX file
3. Wait for processing and matching

#### Step 2: Review the Results
- **Master DB Matches:** See responses in "Deviations" column (auto-filled)
- **PostgreSQL Matches:** Empty "Deviations" - these have historical comments but no Master DB response
- **No Matches:** Empty "Deviations" - brand new requirements

#### Step 3: Add Your Responses
1. Click on any empty cell in the "Deviations" column
2. Type your response (up to 500 characters)
3. Press Enter or click elsewhere to save
4. Repeat for other requirements

#### Step 4: Export Your Work
1. Look for the success message: "✅ X deviations edited/added"
2. Click "📥 Download Results as CSV"
3. File saved as: `requirement_matching_[filename].csv`

### Example Use Cases

#### Use Case 1: Filling Gaps
```
Requirement: "Emergency stop buttons shall be provided"
Has Match: No
Action: Click "Deviations" cell → Type your response
Response: "Red mushroom head e-stop buttons will be provided at 4 locations as per layout drawing"
```

#### Use Case 2: Updating PostgreSQL Matches
```
Requirement: "Training records shall be maintained for 5 years"
Has Match: Yes - PostgreSQL (but no Master DB response)
Action: Click "Deviations" cell → Add your standardized response
Response: "Training records maintained in TrainingManager software with automated archival after 5 years"
```

#### Use Case 3: Reviewing Master DB Responses
```
Requirement: "CIP System with 4 phases"
Has Match: Yes - Master DB
Deviations Column: "WIP System will be provided with pre-rinse, caustic wash, intermediate rinse, and sanitization phases"
Action: Review for accuracy (no editing needed if correct)
```

### Export Format

The CSV export includes all columns:

```csv
New Requirement,Category,Priority,Deviations,Has Match,Match Source,Similarity Score,...
"CIP System","Cleaning","High","WIP System will be provided","Yes - Master DB","Master DB (DEV-0001)","1.00",...
"Training records","Data","Medium","Training records maintained in software","Yes - PostgreSQL","Doc123.docx","0.85",...
"New requirement","Process","Low","Custom response added by user","No","-","0.00",...
```

### Benefits

1. ✅ **Efficiency:** Pre-filled Master DB responses save time
2. ✅ **Completeness:** Add responses for new requirements on the spot
3. ✅ **Traceability:** Export shows which responses came from Master DB vs. user-added
4. ✅ **Flexibility:** Edit any empty cell as needed
5. ✅ **Documentation:** Download CSV for records and sharing

### Tips

💡 **Tip 1:** Master DB responses are shown but locked to preserve original data integrity

💡 **Tip 2:** The edit counter shows how many deviations you've added/modified

💡 **Tip 3:** You can export at any time - even with partial edits

💡 **Tip 4:** Use concise, clear language in your responses (500 char limit)

💡 **Tip 5:** The CSV preserves all matching metadata (similarity scores, sources, etc.)

### Technical Details

- **Component:** `st.data_editor` (Streamlit interactive table)
- **Disabled Columns:** All except "Deviations"
- **Column Type:** TextColumn with 500 character limit
- **Export Format:** UTF-8 encoded CSV
- **Auto-save:** Changes saved immediately on cell exit

### Next Steps

After exporting your CSV, you can:
- Import into Master Database Excel file
- Share with team for review
- Use as input for deviation tracking systems
- Archive as part of project documentation
