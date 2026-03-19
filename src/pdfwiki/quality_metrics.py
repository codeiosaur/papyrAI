"""
Quality metrics for wiki generation.

Measure fact accuracy, wikilink integrity, and Obsidian compatibility.
Use this to evaluate model/prompt optimizations consistently.
"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WikiPageQuality:
    """Quality metrics for a generated wiki page."""
    filename: str
    concept: str
    
    # Structure metrics
    has_filename_marker: bool  # "FILENAME: ..." present
    has_related_section: bool
    section_count: int
    wikilink_count: int
    
    # Link integrity
    orphaned_links: list[str]  # [[links]] not in valid_names
    valid_link_count: int
    link_coverage: float  # percent of valid names mentioned
    
    # Obsidian validation
    syntax_errors: list[str]
    is_valid_markdown: bool
    
    # Fact check
    potential_hallucinations: list[str]  # facts potentially not in source


def validate_wiki_page(
    page_content: str,
    concept: str,
    valid_wikilink_names: list[str],
    source_text: str = "",
) -> WikiPageQuality:
    """
    Analyze a generated wiki page for quality issues.
    
    Args:
        page_content: Full markdown page content
        concept: The concept this page is about
        valid_wikilink_names: List of valid [[link]] targets
        source_text: Original source material (for hallucination detection)
    
    Returns:
        WikiPageQuality with metrics and issues
    """
    
    # Extract filename marker
    filename_match = re.match(r'^FILENAME:\s*(.+?)$', page_content, re.MULTILINE)
    has_filename_marker = bool(filename_match)
    extracted_filename = filename_match.group(1).strip() if filename_match else ""
    
    # Count sections
    section_pattern = r'^#{2,4}\s'
    sections = re.findall(section_pattern, page_content, re.MULTILINE)
    section_count = len(sections)
    has_related = bool(re.search(r'^###\s+Related', page_content, re.MULTILINE))
    
    # Extract and validate wikilinks
    wikilink_pattern = r'\[\[([^\]]+)\]\]'
    found_links = re.findall(wikilink_pattern, page_content)
    wikilink_count = len(found_links)
    
    # Check for orphaned links (not in valid names)
    normalized_valid = {name.lower().strip() for name in valid_wikilink_names}
    orphaned = [
        link for link in found_links
        if link.lower().strip() not in normalized_valid
    ]
    valid_link_count = wikilink_count - len(orphaned)
    
    # Link coverage: did we mention enough valid concepts?
    valid_mentioned = {
        name for name in valid_wikilink_names
        if name.lower() in page_content.lower()
    }
    link_coverage = (len(valid_mentioned) / len(valid_wikilink_names) * 100) if valid_wikilink_names else 0
    
    # Syntax validation
    syntax_errors = []
    
    # Check for markdown syntax issues
    # Unmatched brackets
    open_brackets = page_content.count('[') - page_content.count('[[') * 2
    close_brackets = page_content.count(']') - page_content.count(']]') * 2
    if open_brackets != close_brackets:
        syntax_errors.append(f"Unmatched brackets (open: {open_brackets}, close: {close_brackets})")
    
    # Check for malformed wikilinks
    malformed_wikilinks = re.findall(r'\[\[+[^\]]*\]*(?!\])', page_content)
    if malformed_wikilinks:
        syntax_errors.append(f"Potentially malformed wikilinks: {malformed_wikilinks[:3]}")
    
    is_valid_markdown = len(syntax_errors) == 0
    
    # Hallucination detection: simple heuristic
    # Flag sentences that use strong claims but aren't substantiated in source
    hallucination_phrases = []
    if source_text:
        # Look for definitive claims (uses "is", "are", "always", etc.)
        claim_pattern = r'(?:^|\s)(The|A|This)\s+\w+\s+(?:is|are|uses|creates|contains|means)'
        claims = re.finditer(claim_pattern, page_content, re.IGNORECASE)
        
        source_lower = source_text.lower()
        for match in claims:
            sentence_start = max(0, match.start() - 100)
            sentence_end = min(len(page_content), match.end() + 200)
            chunk = page_content[sentence_start:sentence_end]
            
            # Check if core concept/terms are in source
            key_terms = re.findall(r'\b\w{5,}\b', chunk)
            if not any(term.lower() in source_lower for term in key_terms[:2]):
                hallucination_phrases.append(chunk.strip()[:100])
    
    return WikiPageQuality(
        filename=extracted_filename or concept,
        concept=concept,
        has_filename_marker=has_filename_marker,
        has_related_section=has_related,
        section_count=section_count,
        wikilink_count=wikilink_count,
        orphaned_links=orphaned,
        valid_link_count=valid_link_count,
        link_coverage=link_coverage,
        syntax_errors=syntax_errors,
        is_valid_markdown=is_valid_markdown,
        potential_hallucinations=hallucination_phrases,
    )


def format_quality_report(quality: WikiPageQuality) -> str:
    """Format quality metrics as a human-readable report."""
    report = []
    report.append(f"PAGE: {quality.concept}")
    report.append("-" * 50)
    
    report.append("\n[STRUCTURE]")
    report.append(f"  Filename marker: {'✓' if quality.has_filename_marker else '✗'}")
    report.append(f"  Sections: {quality.section_count}")
    report.append(f"  Related section: {'✓' if quality.has_related_section else '✗'}")
    
    report.append("\n[LINKS]")
    report.append(f"  Total wikilinks: {quality.wikilink_count}")
    report.append(f"  Valid links: {quality.valid_link_count}/{quality.wikilink_count}")
    report.append(f"  Link coverage: {quality.link_coverage:.1f}%")
    if quality.orphaned_links:
        report.append(f"  ⚠ Orphaned links: {quality.orphaned_links[:5]}")
    
    report.append("\n[MARKDOWN]")
    report.append(f"  Valid: {'✓' if quality.is_valid_markdown else '✗'}")
    if quality.syntax_errors:
        for err in quality.syntax_errors:
            report.append(f"  ⚠ {err}")
    
    if quality.potential_hallucinations:
        report.append("\n[HALLUCINATION CHECK]")
        report.append(f"  ⚠ {len(quality.potential_hallucinations)} potential unsourced claims:")
        for claim in quality.potential_hallucinations[:3]:
            report.append(f"     - {claim[:60]}...")
    
    return "\n".join(report)


def batch_quality_report(pages: dict[str, str], valid_names: list[str], source_text: str = "") -> str:
    """Generate quality report for multiple pages."""
    report = []
    
    total_pages = len(pages)
    valid_markdown_count = 0
    avg_coverage = 0
    pages_with_orphans = 0
    
    for concept, content in pages.items():
        quality = validate_wiki_page(content, concept, valid_names, source_text)
        if quality.is_valid_markdown:
            valid_markdown_count += 1
        avg_coverage += quality.link_coverage
        if quality.orphaned_links:
            pages_with_orphans += 1
    
    avg_coverage = avg_coverage / total_pages if total_pages > 0 else 0
    
    report.append(f"\n{'=' * 60}")
    report.append(f"BATCH QUALITY REPORT ({total_pages} pages)")
    report.append(f"{'=' * 60}")
    report.append(f"Valid markdown:      {valid_markdown_count}/{total_pages} ({valid_markdown_count/total_pages*100:.0f}%)")
    report.append(f"Avg link coverage:   {avg_coverage:.1f}%")
    report.append(f"Pages with orphaned links: {pages_with_orphans}")
    report.append(f"{'=' * 60}\n")
    
    return "\n".join(report)
