"""
Text chunking and aggregation for RAG system.
"""
import re
import os
from typing import List, Dict, Any, Tuple, Sequence, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dataclasses import dataclass, field
from config import DEFAULT_LLM_MODEL


class CrossPageTextSplitter:
    """Enhanced document chunking with cross-page support."""
    
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 100):
        """Initialize cross-page text splitter."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=DEFAULT_LLM_MODEL,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def split_document(self, document_data: Dict) -> List[Document]:
        """Split document into chunks with format-specific handling."""
        doc_type = document_data['metainfo']['document_type']
        
        if doc_type == 'excel':
            chunks = self._split_excel_document(document_data)
        elif doc_type == 'pdf':
            chunks = self._split_pdf_document(document_data)
        elif doc_type == 'pptx':
            chunks = self._split_pptx_document(document_data)
        else:
            chunks = self._split_cross_page_document(document_data)
        
        return chunks
    
    def _split_cross_page_document(self, document_data: Dict) -> List[Document]:
        """Split document content across page boundaries for better semantic continuity."""
        pages = document_data['content']['pages']
        if not pages:
            return []
        
        combined_text, page_boundaries = self._combine_pages_with_markers(pages)
        if not combined_text.strip():
            return []
        
        text_chunks = self.text_splitter.split_text(combined_text)
        
        documents = []
        for i, chunk in enumerate(text_chunks):
            if not chunk.strip():
                continue
                
            clean_chunk = self._remove_page_markers(chunk)
            chunk_start = combined_text.find(chunk)
            chunk_end = chunk_start + len(chunk)
            page_range = self._get_page_range(chunk_start, chunk_end, page_boundaries)
            
            metadata = {
                "chunk": i + 1,
                "total_chunks": len(text_chunks),
                "content_type": "cross_page_text"
            }
            
            if len(page_range) == 1:
                metadata["page"] = page_range[0]
            else:
                metadata["page"] = page_range[0]
                metadata["page_range"] = ",".join(map(str, page_range))
                metadata["spans_pages"] = True
            
            documents.append(Document(
                page_content=clean_chunk.strip(),
                metadata=metadata
            ))
        
        return documents
    
    def _combine_pages_with_markers(self, pages: List[Dict]) -> Tuple[str, List[Tuple[int, int, int]]]:
        """Combine pages into continuous text with boundary markers."""
        combined_parts = []
        page_boundaries = []
        current_pos = 0
        
        for page_data in pages:
            page_num = page_data['page']
            page_text = page_data.get('text', '').strip()
            
            if not page_text:
                continue
            
            page_marker = f"\n--- PAGE {page_num} ---\n"
            page_content = page_marker + page_text + "\n"
            
            start_pos = current_pos
            end_pos = current_pos + len(page_content)
            
            page_boundaries.append((page_num, start_pos, end_pos))
            combined_parts.append(page_content)
            current_pos = end_pos
        
        return ''.join(combined_parts), page_boundaries
    
    def _remove_page_markers(self, text: str) -> str:
        """Remove page markers from chunk text."""
        return re.sub(r'\n--- PAGE \d+ ---\n', '\n', text)
    
    def _get_page_range(self, chunk_start: int, chunk_end: int, page_boundaries: List[Tuple[int, int, int]]) -> List[int]:
        """Determine which pages a chunk spans based on position."""
        covered_pages = []
        
        for page_num, start_pos, end_pos in page_boundaries:
            if chunk_start < end_pos and chunk_end > start_pos:
                covered_pages.append(page_num)
        
        return sorted(covered_pages) if covered_pages else [1]
    
    def _split_excel_document(self, document_data: Dict) -> List[Document]:
        """Excel-specific chunking: each page (sheet) becomes one complete chunk."""
        chunks = []
        pages = document_data['content']['pages']
        
        for page in pages:
            page_text = page.get('text', '')
            if not page_text.strip():
                continue
            
            # Create one chunk per Excel sheet (page)
            chunk = Document(
                page_content=page_text.strip(),
                metadata={
                    "page": page['page'],
                    "chunk": 1,
                    "total_chunks": 1,
                    "content_type": "excel_full_sheet",
                    "sheet_name": self._extract_sheet_name(page_text),
                    "chunking_strategy": "full_sheet_per_chunk"
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _extract_sheet_name(self, sheet_text: str) -> str:
        """Extract sheet name from Excel sheet text."""
        lines = sheet_text.split('\n')
        for line in lines:
            if line.startswith('Sheet:'):
                sheet_name = line.replace('Sheet:', '').strip()
                return sheet_name if sheet_name else 'Unknown Sheet'
        return 'Unknown Sheet'
    
    def _split_pdf_document(self, document_data: Dict) -> List[Document]:
        """PDF-specific chunking with layout analysis and column detection."""
        chunks = []
        
        for page in document_data['content']['pages']:
            page_text = page.get('text', '')
            if not page_text.strip():
                continue
            
            # Get layout information if available
            layout_type = page.get('layout_type', 'single_column')
            column_count = page.get('column_count', 1)
            extraction_method = page.get('extraction_method', 'standard')
            
            page_chunks = self._split_pdf_page(
                page_text, 
                page['page'], 
                layout_type, 
                column_count,
                extraction_method
            )
            chunks.extend(page_chunks)
        
        return chunks
    
    def _split_pdf_page(self, page_text: str, page_num: int, layout_type: str, 
                       column_count: int, extraction_method: str) -> List[Document]:
        """Split PDF page while preserving layout-aware structure."""
        # Choose chunking strategy based on layout analysis
        if layout_type == 'single_column' or column_count <= 1:
            return self._chunk_single_column_pdf(page_text, page_num, extraction_method)
        elif layout_type == 'multi_column' and column_count > 1:
            return self._chunk_multi_column_pdf(page_text, page_num, column_count)
        else:
            # Complex or unstructured layout
            return self._chunk_complex_pdf(page_text, page_num)
    
    def _chunk_single_column_pdf(self, page_text: str, page_num: int, 
                                 extraction_method: str) -> List[Document]:
        """Handle single-column PDF pages with standard chunking."""
        text_chunks = self.text_splitter.split_text(page_text)
        documents = []
        
        for i, chunk in enumerate(text_chunks):
            if not chunk.strip():
                continue
            
            documents.append(Document(
                page_content=chunk.strip(),
                metadata={
                    "page": page_num,
                    "chunk": i + 1,
                    "total_chunks": len(text_chunks),
                    "content_type": "pdf_single_column",
                    "layout_type": "single_column",
                    "extraction_method": extraction_method
                }
            ))
        
        return documents
    
    def _chunk_multi_column_pdf(self, page_text: str, page_num: int, 
                                column_count: int) -> List[Document]:
        """Handle multi-column PDF layouts with column-aware chunking."""
        documents = []
        
        # Strategy 1: Look for column markers (explicit column separation)
        if '||COLUMN||' in page_text:
            # Split by column markers and process each column independently
            columns = page_text.split('||COLUMN||')
            chunk_counter = 1
            
            for column_idx, column_content in enumerate(columns):
                if not column_content.strip():
                    continue
                
                # Process each column independently with its own chunking
                column_chunks = self.text_splitter.split_text(column_content.strip())
                
                for chunk_idx, chunk in enumerate(column_chunks):
                    if not chunk.strip():
                        continue
                    
                    documents.append(Document(
                        page_content=chunk.strip(),
                        metadata={
                            "page": page_num,
                            "chunk": chunk_counter,
                            "total_chunks": None,  # Will be set later
                            "column": column_idx + 1,
                            "column_chunk": chunk_idx + 1,
                            "content_type": "pdf_multi_column",
                            "layout_type": "multi_column",
                            "column_count": column_count
                        }
                    ))
                    chunk_counter += 1
        else:
            # Strategy 2: Standard chunking for multi-column content without explicit markers
            text_chunks = self.text_splitter.split_text(page_text)
            
            for i, chunk in enumerate(text_chunks):
                if not chunk.strip():
                    continue
                
                documents.append(Document(
                    page_content=chunk.strip(),
                    metadata={
                        "page": page_num,
                        "chunk": i + 1,
                        "total_chunks": len(text_chunks),
                        "content_type": "pdf_multi_column_merged",
                        "layout_type": "multi_column",
                        "column_count": column_count
                    }
                ))
        
        # Update total_chunks for column-based chunking
        total_chunks = len(documents)
        for doc in documents:
            if doc.metadata.get("total_chunks") is None:
                doc.metadata["total_chunks"] = total_chunks
        
        return documents
    
    def _chunk_complex_pdf(self, page_text: str, page_num: int) -> List[Document]:
        """Handle complex PDF layouts with fallback chunking."""
        text_chunks = self.text_splitter.split_text(page_text)
        documents = []
        
        for i, chunk in enumerate(text_chunks):
            if not chunk.strip():
                continue
            
            documents.append(Document(
                page_content=chunk.strip(),
                metadata={
                    "page": page_num,
                    "chunk": i + 1,
                    "total_chunks": len(text_chunks),
                    "content_type": "pdf_complex_layout",
                    "layout_type": "complex",
                    "extraction_method": "fallback"
                }
            ))
        
        return documents
    
    def _split_pptx_document(self, document_data: Dict) -> List[Document]:
        """PPTX-specific chunking: each slide becomes one chunk."""
        chunks = []
        pages = document_data['content']['pages']
        
        for page in pages:
            page_text = page.get('text', '')
            if not page_text.strip():
                continue
            
            # For PPTX, each slide is treated as a complete unit
            chunk = Document(
                page_content=page_text.strip(),
                metadata={
                    "page": page['page'],
                    "chunk": 1,
                    "total_chunks": 1,
                    "content_type": "pptx_full_slide",
                    "slide_number": page['page'],
                    "chunking_strategy": "full_slide_per_chunk"
                }
            )
            chunks.append(chunk)
        
        return chunks


class ParentPageAggregator:
    """Enhanced parent page retrieval with support for cross-page chunks.
    
    Only applies aggregation to PDF documents as Excel and PPTX already
    have one page per chunk and do not need aggregation.
    """
    
    def __init__(self, parsed_reports: List[Dict]):
        """Initialize with parsed document reports."""
        self.parsed_reports = parsed_reports
        self.page_content_map = self._build_page_content_map()
    
    def _build_page_content_map(self) -> Dict[str, Dict[int, str]]:
        """Build mapping from (source_file, page_num) to full page content.
        
        Returns:
            Dict with structure: {source_file: {page_num: page_content}}
        """
        content_map = {}
        for report in self.parsed_reports:
            # Use filename from metainfo as the key
            filename = report['report']['metainfo']['filename']
            content_map[filename] = {}
            
            for page_data in report['report']['content']['pages']:
                page_num = page_data['page']
                page_content = page_data.get('text', '')
                content_map[filename][page_num] = page_content
                
        return content_map
    
    def aggregate_to_parent_pages(self, chunk_results: List[Dict]) -> List[Dict]:
        """Extract parent pages from chunks with intelligent content matching.
        
        Only applies aggregation to PDF chunks. Excel and PPTX chunks are
        returned as-is since they already represent complete pages.
        
        For PDF chunks, this method finds the actual page that contains the chunk content
        by performing content matching, ensuring accurate page attribution in source references.
        """
        processed_chunks = set()
        parent_results = []
        
        for chunk_result in chunk_results:
            document_type = chunk_result.get('document_type', 'unknown')
            
            # Only apply aggregation to PDF documents
            if document_type != 'pdf':
                # For non-PDF documents, return chunk as-is
                parent_results.append(chunk_result)
                continue
            
            # Create a unique identifier for this chunk to avoid duplicates
            chunk_id = (
                chunk_result.get('source_file', ''),
                chunk_result.get('page', 0),
                chunk_result.get('chunk', 0)
            )
            
            if chunk_id in processed_chunks:
                continue
            
            processed_chunks.add(chunk_id)
            
            # For PDF chunks, find the correct page containing the content
            source_file_ref = chunk_result.get('source_file', '')
            filename = self._extract_filename_from_source(source_file_ref)
            chunk_text = chunk_result.get('text', '')
            original_page = chunk_result.get('page', 0)
            
            # Find the best matching page for this chunk content
            best_page, full_page_content = self._find_best_matching_page(filename, chunk_text, original_page)
            
            if full_page_content and full_page_content.strip():
                # Update source reference with correct page number
                correct_source_ref = self._update_source_reference(source_file_ref, best_page)
                
                parent_result = {
                    'text': full_page_content,
                    'page': best_page,  # Use the actual page where content was found
                    'page_range': None,  # Single page
                    'spans_pages': False,
                    'distance': chunk_result['distance'],
                    'source_file': correct_source_ref,
                    'source_reference': correct_source_ref,
                    'document_type': chunk_result['document_type'],
                    'metadata': chunk_result['metadata']
                }
                parent_results.append(parent_result)
            else:
                # Fallback to original chunk if no matching page found
                parent_results.append(chunk_result)
        
        return parent_results
    
    def _extract_filename_from_source(self, source_file_ref: str) -> str:
        """Extract filename from source_file reference.
        
        Args:
            source_file_ref: Source reference like "file.pdf (Page 5)" or "file.pptx (Slide 3)"
            
        Returns:
            Just the filename part, e.g., "file.pdf"
        """
        if not source_file_ref:
            return ''
        
        # Remove page/slide information in parentheses
        import re
        # Match patterns like " (Page X)", " (Slide X)", " Sheet: X"
        cleaned = re.sub(r' \((Page|Slide) \d+\)$', '', source_file_ref)
        cleaned = re.sub(r' Sheet: .+$', '', cleaned)
        
        return cleaned.strip()
    
    def _find_best_matching_page(self, filename: str, chunk_text: str, original_page: int) -> Tuple[int, str]:
        """Find the page that best matches the chunk content using intelligent content matching.
        
        Args:
            filename: The filename to search within
            chunk_text: The text content of the chunk to match
            original_page: The originally attributed page number
            
        Returns:
            Tuple of (best_page_number, full_page_content)
        """
        if filename not in self.page_content_map:
            return original_page, ''
        
        file_pages = self.page_content_map[filename]
        
        # Clean the chunk text for better matching
        clean_chunk = self._clean_text_for_matching(chunk_text)
        
        if not clean_chunk.strip():
            # If chunk text is empty, fall back to original page
            return original_page, file_pages.get(original_page, '')
        
        # Strategy 1: Exact substring match - highest confidence
        for page_num, page_content in file_pages.items():
            clean_page = self._clean_text_for_matching(page_content)
            if clean_chunk in clean_page and len(clean_chunk) > 50:  # Minimum meaningful length
                return page_num, page_content
        
        # Strategy 2: Significant overlap match - medium confidence
        best_match_page = original_page
        best_match_content = file_pages.get(original_page, '')
        best_overlap_ratio = 0.0
        
        for page_num, page_content in file_pages.items():
            overlap_ratio = self._calculate_content_overlap(clean_chunk, self._clean_text_for_matching(page_content))
            if overlap_ratio > best_overlap_ratio and overlap_ratio > 0.3:  # At least 30% overlap
                best_overlap_ratio = overlap_ratio
                best_match_page = page_num
                best_match_content = page_content
        
        # Strategy 3: Key phrase matching for critical content
        if best_overlap_ratio < 0.3:
            key_phrases = self._extract_key_phrases(clean_chunk)
            if key_phrases:
                for page_num, page_content in file_pages.items():
                    clean_page = self._clean_text_for_matching(page_content)
                    matched_phrases = sum(1 for phrase in key_phrases if phrase in clean_page)
                    match_ratio = matched_phrases / len(key_phrases)
                    
                    if match_ratio > 0.5:  # At least 50% of key phrases match
                        return page_num, page_content
        
        return best_match_page, best_match_content
    
    def _clean_text_for_matching(self, text: str) -> str:
        """Clean text for better content matching by normalizing whitespace and removing noise.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text suitable for content matching
        """
        if not text:
            return ''
        
        import re
        
        # Normalize whitespace and remove excessive spacing
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common formatting artifacts that might interfere with matching
        cleaned = re.sub(r'[\u00a0\u2000-\u200f\u2028-\u202f\ufeff]', ' ', cleaned)  # Unicode spaces
        cleaned = re.sub(r'[^\w\s\u4e00-\u9fff\u3400-\u4dbf]', ' ', cleaned)  # Keep words, spaces, and Chinese characters
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _calculate_content_overlap(self, text1: str, text2: str) -> float:
        """Calculate content overlap ratio between two texts using word-level comparison.
        
        Args:
            text1: First text (usually chunk content)
            text2: Second text (usually page content)
            
        Returns:
            Overlap ratio from 0.0 to 1.0
        """
        if not text1 or not text2:
            return 0.0
        
        # Split into words and create sets for comparison
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1:
            return 0.0
        
        # Calculate overlap ratio based on words in text1 that appear in text2
        common_words = words1.intersection(words2)
        overlap_ratio = len(common_words) / len(words1)
        
        return overlap_ratio
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text for content matching.
        
        Args:
            text: Text to extract key phrases from
            
        Returns:
            List of key phrases that can be used for matching
        """
        if not text or len(text) < 20:
            return []
        
        import re
        
        key_phrases = []
        
        # Extract phrases with specific patterns that are likely to be unique
        # Pattern 1: Sequences of 3+ consecutive words containing Chinese characters or important terms
        chinese_phrases = re.findall(r'[\u4e00-\u9fff]+[^.!?]*?[\u4e00-\u9fff]+', text)
        for phrase in chinese_phrases:
            if len(phrase.strip()) > 10:  # Meaningful length
                key_phrases.append(phrase.strip()[:50])  # Limit length
        
        # Pattern 2: Number sequences (important for regulations, financial data, etc.)
        number_phrases = re.findall(r'\d+[^\w]*[^\d\s]+[^\w]*\d+', text)
        key_phrases.extend(number_phrases)
        
        # Pattern 3: Capitalized sequences (might be important terms, names, etc.)
        caps_phrases = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', text)
        key_phrases.extend(caps_phrases)
        
        # Remove duplicates and filter by length
        unique_phrases = list(set(phrase for phrase in key_phrases if len(phrase.strip()) > 5))
        
        # Return top 10 most distinctive phrases
        return unique_phrases[:10]
    
    def _update_source_reference(self, original_source_ref: str, correct_page: int) -> str:
        """Update source reference with correct page number.
        
        Args:
            original_source_ref: Original source reference like "file.pdf (Page 5)"
            correct_page: Correct page number to use
            
        Returns:
            Updated source reference with correct page number
        """
        if not original_source_ref:
            return f"Unknown Document (Page {correct_page})"
        
        import re
        
        # Extract the base filename
        filename = self._extract_filename_from_source(original_source_ref)
        
        # Determine document type and create appropriate reference
        if filename.endswith('.pdf'):
            return f"{filename} (Page {correct_page})"
        elif filename.endswith('.pptx'):
            return f"{filename} (Slide {correct_page})"
        elif filename.endswith('.xls') or filename.endswith('.xlsx'):
            return f"{filename} (Page {correct_page})"
        else:
            return f"{filename} (Page {correct_page})"
    
    def _get_full_page_content(self, filename: str, page_number: int) -> str:
        """Get full content for a specific page from a specific document.
        
        Args:
            filename: The filename to look up
            page_number: The page number to retrieve
            
        Returns:
            Full page content or empty string if not found
        """
        if filename not in self.page_content_map:
            return ''
        
        file_pages = self.page_content_map[filename]
        return file_pages.get(page_number, '')
    
    def _get_chunk_page_coverage(self, chunk_result: Dict) -> List[int]:
        """Determine which pages a chunk covers."""
        metadata = chunk_result.get('metadata', {})
        
        if metadata.get('spans_pages', False) and 'page_range' in metadata:
            page_range_str = metadata['page_range']
            return [int(page) for page in page_range_str.split(',')]
        else:
            return [chunk_result['page']]
    
    def _get_combined_page_content(self, source_file: str, page_numbers: List[int]) -> str:
        """Combine content from multiple pages of a specific document.
        
        Args:
            source_file: The source file identifier
            page_numbers: List of page numbers to combine
            
        Returns:
            Combined page content or empty string if not found
        """
        if source_file not in self.page_content_map:
            return ''
        
        file_pages = self.page_content_map[source_file]
        
        if len(page_numbers) == 1:
            return file_pages.get(page_numbers[0], '')
        
        combined_parts = []
        for page_num in sorted(page_numbers):
            page_content = file_pages.get(page_num, '')
            if page_content.strip():
                combined_parts.append(f"[Page {page_num}]\n{page_content}")
        
        return '\n\n--- PAGE BREAK ---\n\n'.join(combined_parts)
