"""
Replace the matching section in main_gemini.py with this enhanced version
"""

# Add this import at the top of main_gemini.py
from enhanced_matching import match_requirements_enhanced, get_enhanced_matcher

# Replace the matching section (around line 550-650) with this:

if processing_mode == "🔍 Extract + Match Requirements":
    st.divider()
    st.subheader("📊 Enhanced Requirement Matching with Historical Data")
    
    with st.spinner("🔍 Performing intelligent multi-layer matching..."):
        # Initialize enhanced matcher
        enhanced_matcher = get_enhanced_matcher()
        
        # LAYER 1: Check Master Database first (Excel)
        master_db_matches = {}
        if master_db:
            st.info("🔍 **Layer 1:** Checking Master Database (Excel)...")
            for req in requirements:
                # Use enhanced matching with stricter threshold for Master DB
                match = master_db.search_requirement(req['text'], threshold=0.75)
                if match:
                    master_db_matches[req['text']] = match
            
            if master_db_matches:
                st.success(f"✅ Found {len(master_db_matches)} matches in Master Database!")
            else:
                st.info("ℹ️ No matches found in Master Database")
        
        # LAYER 2: Enhanced Historical PostgreSQL Database Matching
        st.info("🔍 **Layer 2:** Performing enhanced semantic matching on PostgreSQL...")
        
        # Get ALL requirements from database for comparison
        all_requirements = vectorstore.get_all_requirements()
        
        # Filter out current file's requirements
        historical_requirements = [
            req for req in all_requirements 
            if req.get('document_name', '') != filename
        ]
        
        st.info(f"Analyzing against {len(historical_requirements)} historical requirements")
        
        # Create matching table data with enhanced matching
        matching_data = []
        
        # Progress bar for matching
        progress_bar = st.progress(0)
        
        for req_idx, req in enumerate(requirements):
            try:
                req_text = req['text']
                
                # Update progress
                progress_bar.progress((req_idx + 1) / len(requirements))
                
                # Check Master DB first (priority)
                if req_text in master_db_matches:
                    master_match = master_db_matches[req_text]
                    
                    matching_data.append({
                        'New Requirement': req_text[:200] + '...' if len(req_text) > 200 else req_text,
                        'Category': req.get('category', 'Unknown'),
                        'Priority': req.get('priority', 'Unknown'),
                        'Matched Requirement': master_match['requirement'][:200] + '...' if len(master_match['requirement']) > 200 else master_match['requirement'],
                        'Match Source': f"Master DB ({master_match['deviation_id']})",
                        'Historical Comments': master_match['response'][:150] + '...' if len(master_match['response']) > 150 else master_match['response'],
                        'Historical Responses': master_match['response'][:150] + '...' if len(master_match['response']) > 150 else master_match['response'],
                        'Similarity Score': f"{master_match['similarity']:.2f}",
                        'Has Match': 'Yes - Master DB',
                        'Match Type': master_match['match_type'],
                        'Confidence': 'high',
                        'Keyword Overlap': 'N/A'
                    })
                    continue
                
                # Enhanced PostgreSQL matching
                # Use enhanced matching with comprehensive validation
                # Threshold 0.6 for high-quality matches only
                matches = match_requirements_enhanced(
                    req_text,
                    historical_requirements,
                    min_score=0.6  # Higher threshold for better quality
                )
                
                if matches:
                    # Take the best match
                    best_match = matches[0]
                    
                    # Extract clean comments if available
                    matched_comments = ""
                    if best_match.get('comments'):
                        clean_comments = extract_clean_comments(best_match['comments'])
                        if clean_comments:
                            comment_texts = []
                            for comment in clean_comments[:3]:
                                text = comment.get('text', '')
                                author = comment.get('author', 'Unknown')
                                if text:
                                    comment_texts.append(f"[{author}] {text}")
                            matched_comments = " | ".join(comment_texts)
                    
                    matching_data.append({
                        'New Requirement': req_text[:200] + '...' if len(req_text) > 200 else req_text,
                        'Category': req.get('category', 'Unknown'),
                        'Priority': req.get('priority', 'Unknown'),
                        'Matched Requirement': best_match['requirement'][:200] + '...' if len(best_match['requirement']) > 200 else best_match['requirement'],
                        'Match Source': best_match['document_name'],
                        'Historical Comments': matched_comments or 'No comments',
                        'Historical Responses': matched_comments or 'No responses',
                        'Similarity Score': f"{best_match['match_score']:.2f}",
                        'Has Match': 'Yes - PostgreSQL',
                        'Match Type': 'enhanced_semantic',
                        'Confidence': best_match['confidence'],
                        'Keyword Overlap': str(best_match['keyword_overlap'])
                    })
                else:
                    # No match found
                    matching_data.append({
                        'New Requirement': req_text[:200] + '...' if len(req_text) > 200 else req_text,
                        'Category': req.get('category', 'Unknown'),
                        'Priority': req.get('priority', 'Unknown'),
                        'Matched Requirement': 'No historical match found',
                        'Match Source': '-',
                        'Historical Comments': '-',
                        'Historical Responses': '-',
                        'Similarity Score': '0.00',
                        'Has Match': 'No',
                        'Match Type': '-',
                        'Confidence': '-',
                        'Keyword Overlap': '0'
                    })
                
            except Exception as e:
                st.error(f"Error matching requirement {req_idx}: {str(e)}")
                matching_data.append({
                    'New Requirement': req['text'],
                    'Category': req.get('category', 'Unknown'),
                    'Priority': req.get('priority', 'Unknown'),
                    'Matched Requirement': f'Error: {str(e)[:50]}',
                    'Match Source': 'Error',
                    'Similarity Score': '0.00',
                    'Has Match': 'Error',
                    'Confidence': 'error',
                    'Keyword Overlap': '0'
                })
        
        progress_bar.empty()
    
    # Display enhanced matching results
    if matching_data:
        import pandas as pd
        
        # Enhanced summary statistics
        total_requirements = len(matching_data)
        master_db_matched = len([x for x in matching_data if x.get('Has Match', '').startswith('Yes - Master DB')])
        postgres_matched = len([x for x in matching_data if x.get('Has Match', '') == 'Yes - PostgreSQL'])
        high_confidence = len([x for x in matching_data if x.get('Confidence', '') == 'high'])
        medium_confidence = len([x for x in matching_data if x.get('Confidence', '') == 'medium'])
        no_match = len([x for x in matching_data if x.get('Has Match', '') == 'No'])
        
        # Display enhanced metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("📝 Total Requirements", total_requirements)
        with col2:
            st.metric("📋 Master DB", master_db_matched, help="Excel Master Database matches")
        with col3:
            st.metric("🗄️ Historical DB", postgres_matched, help="PostgreSQL matches")
        with col4:
            st.metric("✅ High Confidence", high_confidence, help="Matches with high confidence")
        with col5:
            st.metric("❌ No Match", no_match, help="No historical match found")
        
        # Display confidence breakdown
        with st.expander("📊 Matching Quality Metrics", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("High Confidence", high_confidence)
            with col2:
                st.metric("Medium Confidence", medium_confidence)
            with col3:
                low_confidence = postgres_matched - high_confidence - medium_confidence
                st.metric("Low Confidence", max(0, low_confidence))
        
        st.subheader("📊 Enhanced Multi-Layer Requirement Matching Results")
        st.caption("✅ Using advanced semantic matching with keyword validation | 📝 Edit 'Deviations' to add responses")
        
        # Create display DataFrame
        display_df = pd.DataFrame(matching_data)
        
        # Add editable Deviations column
        display_df['Deviations'] = display_df.apply(
            lambda row: row.get('Historical Responses', '') if row.get('Has Match', '').startswith('Yes - Master DB') else '',
            axis=1
        )
        
        # Reorder columns
        column_order = [
            'New Requirement', 'Category', 'Priority',
            'Deviations',
            'Has Match', 'Confidence', 'Keyword Overlap',
            'Match Source', 'Match Type', 'Similarity Score',
            'Matched Requirement', 'Historical Comments', 'Historical Responses'
        ]
        
        column_order = [col for col in column_order if col in display_df.columns]
        display_df = display_df[column_order]
        
        # Display editable table
        edited_df = st.data_editor(
            display_df,
            width='stretch',
            hide_index=True,
            column_config={
                "Deviations": st.column_config.TextColumn(
                    "Deviations / Response",
                    help="Add your response here",
                    max_chars=1000,
                    width="large"
                ),
                "Confidence": st.column_config.TextColumn(
                    "Confidence",
                    help="Match confidence: high (>75%), medium (50-75%), low (<50%)",
                    width="small"
                ),
                "Keyword Overlap": st.column_config.TextColumn(
                    "Keywords",
                    help="Number of common keywords found",
                    width="small"
                ),
                "New Requirement": st.column_config.TextColumn(width="large"),
                "Historical Comments": st.column_config.TextColumn(width="large"),
                "Matched Requirement": st.column_config.TextColumn(width="large")
            },
            disabled=[col for col in display_df.columns if col != 'Deviations']
        )
        
        # Export functionality
        st.divider()
        col_export1, col_export2 = st.columns([3, 1])
        
        with col_export2:
            user_added = edited_df[edited_df['Deviations'] != display_df['Deviations']]
            if not user_added.empty:
                st.success(f"✅ {len(user_added)} responses added")
            
            csv_data = edited_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name=f"enhanced_matching_{filename.replace('.docx', '')}.csv",
                mime="text/csv"
            )
        
        with col_export1:
            if not user_added.empty:
                st.info("💡 Your responses are ready to export")
            else:
                st.info("💡 Add responses in 'Deviations' column then download")
