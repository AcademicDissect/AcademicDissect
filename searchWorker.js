/* eslint-disable no-restricted-globals */
// This worker handles paper searching in a separate thread

/**
 * 智能模糊匹配函数 - 根据字符串特性自动选择最优算法
 *
 * 特性:
 * - 自动根据字符串长度和特性选择最佳算法
 * - 使用优化版本处理长字符串，使用标准版本处理短字符串或可能完全匹配的情况
 * - 对于空字符串、null、undefined或长度小于5的字符串直接返回0
 * - 返回0-100之间的匹配分数
 *
 * @param {string|null|undefined} a 第一个字符串
 * @param {string|null|undefined} b 第二个字符串
 * @returns {number} 匹配分数 (0-100)
 */
function smartFuzzyMatch(a, b) {
  // 处理null、undefined或空字符串的情况
  if (a == null || b == null || a === "" || b === "") {
    return 0;
  }
  // 转换为小写进行不区分大小写的匹配
  const aLower = a.toLowerCase().trim();
  const bLower = b.toLowerCase().trim();
  if (aLower === "" || bLower === "") {
    return 0;
  }
  // 确定哪个字符串较短，用作匹配字符串
  let matchStr;
  let targetStr;
  if (aLower.length <= bLower.length) {
    matchStr = aLower;
    targetStr = bLower;
  } else {
    matchStr = bLower;
    targetStr = aLower;
  }
  // 边缘情况: 空字符串或字符串短于5个字符
  if (!matchStr || matchStr.length < 5) {
    return 0;
  }
  // 快速路径: 完全匹配 (在长度检查之后)
  if (matchStr === targetStr) {
    return 100;
  }
  // 快速路径: 如果短字符串是长字符串的一部分
  if (targetStr.includes(matchStr)) {
    return 100;
  }
  // 决定使用哪个算法 - 根据测试结果选择最佳策略
  const useOptimized =
    matchStr.length > 500 && matchStr.length / targetStr.length > 0.5;
  return useOptimized
    ? fuzzyMatchOptimized(matchStr, targetStr)
    : fuzzyMatchStandard(matchStr, targetStr);
}

/**
 * 标准模糊匹配实现 - 适用于较短的字符串或可能高度匹配的情况
 *
 * @param {string} matchStr 较短的字符串用于匹配
 * @param {string} targetStr 目标字符串
 * @returns {number} 匹配分数 (0-100)
 */
function fuzzyMatchStandard(matchStr, targetStr) {
  // 计算不同长度子串的分数
  let totalScore = 0;
  let maxPossibleScore = 0;
  // 根据子串长度相对于完整字符串长度计算权重
  const calculateWeight = (substringLength) => {
    // 指数权重，使得较长的匹配具有显著更高的价值
    return Math.pow(substringLength / matchStr.length, 2) * 100;
  };
  // 尝试用逐渐缩小的子串进行匹配
  for (let len = matchStr.length; len >= 5; len--) {
    // 当前长度子串的权重
    const weight = calculateWeight(len);
    maxPossibleScore += weight;
    let matchFound = false;
    // 在matchStr中滑动窗口
    const slidingWindowCount = matchStr.length - len + 1;
    for (let i = 0; i < slidingWindowCount; i++) {
      const substring = matchStr.substring(i, i + len);
      if (targetStr.includes(substring)) {
        totalScore += weight;
        matchFound = true;
        break; // 每个长度只计算一次匹配，避免过度加权
      }
    }
    // 如果在当前长度没有找到匹配，尝试较短的长度
    if (!matchFound && len > 5) {
      // 我们会尝试略短的长度(优化性能)
      const decreaseStep = Math.max(1, Math.floor(len / 10));
      len -= decreaseStep - 1; // -1是因为for循环也会减1
    }
  }
  // 计算最终分数，为最大可能分数的百分比
  const finalScore = Math.round((totalScore / maxPossibleScore) * 100);
  return finalScore;
}

/**
 * 优化的模糊匹配实现 - 适用于较长的字符串
 * 使用KMP算法和策略性采样以提高性能
 *
 * @param {string} matchStr 较短的字符串用于匹配
 * @param {string} targetStr 目标字符串
 * @returns {number} 匹配分数 (0-100)
 */
function fuzzyMatchOptimized(matchStr, targetStr) {
  // KMP匹配模式准备
  function buildPatternTable(pattern) {
    const table = [0];
    let prefixIndex = 0;
    let suffixIndex = 1;
    while (suffixIndex < pattern.length) {
      if (pattern[prefixIndex] === pattern[suffixIndex]) {
        table[suffixIndex] = prefixIndex + 1;
        suffixIndex++;
        prefixIndex++;
      } else if (prefixIndex === 0) {
        table[suffixIndex] = 0;
        suffixIndex++;
      } else {
        prefixIndex = table[prefixIndex - 1];
      }
    }
    return table;
  }

  function findFirstOccurrence(pattern, text) {
    if (pattern.length > text.length) return false;
    const patternTable = buildPatternTable(pattern);
    let textIndex = 0;
    let patternIndex = 0;
    while (textIndex < text.length) {
      if (pattern[patternIndex] === text[textIndex]) {
        patternIndex++;
        textIndex++;

        if (patternIndex === pattern.length) {
          return true; // 找到匹配
        }
      } else if (patternIndex === 0) {
        textIndex++;
      } else {
        patternIndex = patternTable[patternIndex - 1];
      }
    }
    return false; // 未找到匹配
  }
  // 计算子串分数
  let totalScore = 0;
  let maxPossibleScore = 0;
  // 根据子串长度相对于完整字符串长度计算权重
  const calculateWeight = (substringLength) => {
    return Math.pow(substringLength / matchStr.length, 2) * 100;
  };
  // 尝试用逐渐缩小的子串进行匹配
  for (let len = matchStr.length; len >= 5; len--) {
    const weight = calculateWeight(len);
    maxPossibleScore += weight;
    let matchFound = false;
    // 策略性检查少量位置而不是每个可能的位置
    const slidingWindowCount = matchStr.length - len + 1;
    const checkPositions = Math.min(slidingWindowCount, 5); // 最多检查5个位置
    const step = Math.max(1, Math.floor(slidingWindowCount / checkPositions));
    for (let i = 0; i < slidingWindowCount; i += step) {
      const substring = matchStr.substring(i, i + len);
      if (findFirstOccurrence(substring, targetStr)) {
        totalScore += weight;
        matchFound = true;
        break;
      }
    }
    // 如果在当前长度没有找到匹配，跳过一些长度以提高性能
    if (!matchFound && len > 5) {
      const decreaseStep = Math.max(1, Math.floor(len / 8));
      len -= decreaseStep - 1;
    }
  }
  const finalScore = Math.round((totalScore / maxPossibleScore) * 100);
  return finalScore;
}

// Calculate match score for a paper based on search terms
const calculateMatchScore = (paper, searchTerms, fullSearchPhrase) => {
  let score = 0;
  let hasExactMatch = false;

  // Check for exact matches (100% match case)
  const searchTermTrimmed = fullSearchPhrase.trim().toLowerCase();
  if (searchTermTrimmed && searchTermTrimmed.length > 0) {
    // Check exact title match
    if (paper?.Title?.toLowerCase() === searchTermTrimmed) {
      hasExactMatch = true;
    }

    // Check exact tag match
    if (paper?.Tags?.some((tag) => tag.toLowerCase() === searchTermTrimmed)) {
      hasExactMatch = true;
    }

    // Check exact author match
    const authors = paper?.Authors?.toLowerCase().split(" · ");
    if (authors.includes(searchTermTrimmed)) {
      hasExactMatch = true;
    }

    // For individual search terms
    if (searchTerms.length === 1 && searchTerms[0].trim().length > 0) {
      const term = searchTerms[0].trim().toLowerCase();

      // Check if term exactly matches any field
      if (
        paper?.Title?.toLowerCase() === term ||
        paper?.Tags?.some((tag) => tag.toLowerCase() === term) ||
        authors.includes(term) ||
        paper?.Abstract?.toLowerCase() === term ||
        paper?.Abstract_Summary?.toLowerCase() === term
      ) {
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
  searchTerms.forEach((term) => {
    if (!term.trim()) return;

    const titleLower = paper.Title.toLowerCase();
    if (titleLower.includes(term)) {
      // Higher score for matches at word boundaries
      if (new RegExp(`\\b${term}\\b`, "i").test(titleLower)) {
        score += 40;
      } else {
        score += 30;
      }
    }
  });

  // Tag matches (high weight)
  paper?.Tags?.forEach((tag) => {
    searchTerms?.forEach((term) => {
      if (!term.trim()) return;

      const tagLower = tag.toLowerCase();
      if (tagLower.includes(term)) {
        // Higher score for matches at word boundaries
        if (new RegExp(`\\b${term}\\b`, "i").test(tagLower)) {
          score += 20;
        } else {
          score += 15;
        }
      }
    });
  });

  // Author matches
  searchTerms.forEach((term) => {
    if (!term.trim()) return;

    if (paper.Authors.toLowerCase().includes(term)) {
      score += 10;
    }
  });

  // Abstract matches
  searchTerms.forEach((term) => {
    if (!term.trim()) return;

    if (paper.Abstract.toLowerCase().includes(term)) {
      // Count occurrences for more relevant matches
      const count = (
        paper.Abstract.toLowerCase().match(new RegExp(term, "g")) || []
      ).length;
      score += 5 + Math.min(count, 5); // Cap at 5 additional points
    }
  });

  return score;
};

// Filter and score papers
const filterAndScorePapers = (papers, searchTerms, fullSearchPhrase) => {
  return papers
    .map((paper) => {
      const paperTitle = paper.Title?.toLowerCase();
      const paperAbstract = paper.Abstract?.toLowerCase();
      const paperSummary = paper.Abstract_Summary?.toLowerCase();
      const paperAuthors = paper.Authors?.toLowerCase();
      const paperTags = paper?.Tags?.map((tag) => tag?.toLowerCase()) ?? [];
      const titleMatch = smartFuzzyMatch(fullSearchPhrase, paperTitle);
      const abstractMatch = smartFuzzyMatch(fullSearchPhrase, paperAbstract);
      const summaryMatch = smartFuzzyMatch(fullSearchPhrase, paperSummary);
      const authorsMatch = smartFuzzyMatch(fullSearchPhrase, paperAuthors);
      const tagMatch =
        paperTags
          .map((i) => smartFuzzyMatch(fullSearchPhrase, i))
          .sort((a, b) => b - a)[0] ?? 0;
      const max =
        [titleMatch, abstractMatch, summaryMatch, authorsMatch].sort(
          (a, b) => b - a
        )[0] ?? 0;

      return {
        ...paper,
        matchScore: tagMatch >= 90 ? 0.4 * tagMatch + 0.6 * max : max,
      };
    })
    .filter((i) => i.matchScore > 0)
    .sort((a, b) => b.matchScore - a.matchScore);
};

// Listen for messages from the main thread
self.addEventListener("message", (e) => {
  const { papers, searchTerms, fullSearchPhrase, workerId } = e.data;

  // Process papers in this worker
  const results = filterAndScorePapers(papers, searchTerms, fullSearchPhrase);
  // Send results back to main thread
  self.postMessage({
    results,
    workerId,
  });
});
