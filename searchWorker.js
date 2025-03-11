/* eslint-disable no-restricted-globals */
// This worker handles paper searching in a separate thread

// Calculate match score for a paper based on search terms
const calculateMatchScore = (paper, searchTerms, fullSearchPhrase) => {
    let score = 0;
    let hasExactMatch = false;
    
    // Check for exact matches (100% match case)
    const searchTermTrimmed = fullSearchPhrase.trim().toLowerCase();
    if (searchTermTrimmed && searchTermTrimmed.length > 0) {
      // Check exact title match
      if (paper.Title.toLowerCase() === searchTermTrimmed) {
        hasExactMatch = true;
      }
      
      // Check exact tag match
      if (paper.Tags.some(tag => tag.toLowerCase() === searchTermTrimmed)) {
        hasExactMatch = true;
      }
      
      // Check exact author match
      const authors = paper.Authors.toLowerCase().split(' · ');
      if (authors.includes(searchTermTrimmed)) {
        hasExactMatch = true;
      }
      
      // For individual search terms
      if (searchTerms.length === 1 && searchTerms[0].trim().length > 0) {
        const term = searchTerms[0].trim().toLowerCase();
        
        // Check if term exactly matches any field
        if (paper.Title.toLowerCase() === term ||
            paper.Tags.some(tag => tag.toLowerCase() === term) ||
            authors.includes(term) ||
            paper.Abstract.toLowerCase() === term ||
            paper.Abstract_Summary.toLowerCase() === term) {
          hasExactMatch = true;
        }
      }
    }
    
    // If there's an exact match, return maximum score
    if (hasExactMatch) {
      return 150; // This will normalize to 100% in formatMatchScore
    }
    
    // Otherwise calculate partial match score
    
    // Title matches (highest weight)
    searchTerms.forEach(term => {
      if (!term.trim()) return;
      
      const titleLower = paper.Title.toLowerCase();
      if (titleLower.includes(term)) {
        // Higher score for matches at word boundaries
        if (new RegExp(`\\b${term}\\b`, 'i').test(titleLower)) {
          score += 40;
        } else {
          score += 30;
        }
      }
    });
    
    // Tag matches (high weight)
    paper.Tags.forEach(tag => {
      searchTerms.forEach(term => {
        if (!term.trim()) return;
        
        const tagLower = tag.toLowerCase();
        if (tagLower.includes(term)) {
          // Higher score for matches at word boundaries
          if (new RegExp(`\\b${term}\\b`, 'i').test(tagLower)) {
            score += 20;
          } else {
            score += 15;
          }
        }
      });
    });
    
    // Author matches
    searchTerms.forEach(term => {
      if (!term.trim()) return;
      
      if (paper.Authors.toLowerCase().includes(term)) {
        score += 10;
      }
    });
    
    // Abstract matches
    searchTerms.forEach(term => {
      if (!term.trim()) return;
      
      if (paper.Abstract.toLowerCase().includes(term)) {
        // Count occurrences for more relevant matches
        const count = (paper.Abstract.toLowerCase().match(new RegExp(term, 'g')) || []).length;
        score += 5 + Math.min(count, 5); // Cap at 5 additional points
      }
    });
    
    return score;
  };
  
  // Filter and score papers
  const filterAndScorePapers = (papers, searchTerms, fullSearchPhrase) => {
    return papers
      .filter(paper => {
        // Check for full phrase match in any field
        const paperTitle = paper?.Title?.toLowerCase();
        const paperAbstract = paper?.Abstract?.toLowerCase();
        const paperSummary = paper?.Abstract_Summary?.toLowerCase();
        const paperAuthors = paper?.Authors?.toLowerCase();
        const paperTags = paper.Tags.map(tag => tag.toLowerCase());
        
        // Full phrase match
        if (paperTitle.includes(fullSearchPhrase) || 
            paperAbstract.includes(fullSearchPhrase) || 
            paperSummary.includes(fullSearchPhrase) || 
            paperAuthors.includes(fullSearchPhrase) || 
            paperTags.some(tag => tag.includes(fullSearchPhrase))) {
          return true;
        }
        
        // Individual term matches
        const titleMatch = searchTerms.some(term => paperTitle.includes(term));
        const authorMatch = searchTerms.some(term => paperAuthors.includes(term));
        const abstractMatch = searchTerms.some(term => 
          paperAbstract.includes(term) || paperSummary.includes(term)
        );
        const tagMatch = searchTerms.some(term => 
          paperTags.some(tag => tag.includes(term))
        );
        
        return titleMatch || authorMatch || abstractMatch || tagMatch;
      })
      .map(paper => ({
        ...paper,
        matchScore: calculateMatchScore(paper, searchTerms, fullSearchPhrase)
      }));
  };
  
  // Listen for messages from the main thread
  self.addEventListener('message', (e) => {
    const { papers, searchTerms, fullSearchPhrase, workerId } = e.data;
    
    // Process papers in this worker
    const results = filterAndScorePapers(papers, searchTerms, fullSearchPhrase);
    
    // Send results back to main thread
    self.postMessage({
      results,
      workerId
    });
  });