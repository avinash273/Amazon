--------------------------------------------------------------------------------

All Array Questions

--------------------------------------------------------------------------------
Two Sum (array)
--------------------------------------------------------------------------------
Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums)):
            check = target - nums[i]
            for j in range(len(nums)):
                if((nums[j] == check) and i!=j):
                    return [i,j]
        return None

--------------------------------------------------------------------------------
 Longest Substring Without Repeating Characters
--------------------------------------------------------------------------------
Input: "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.

longest = 0
        left, right = 0, 0
        chars = set()
        while left < len(string) and right < len(string):
            if string[right] not in chars:
                chars.add(string[right])
                right += 1
                longest = max(longest, right - left)
            else:
                chars.remove(string[left])
                left += 1
        return longest

--------------------------------------------------------------------------------
Add Two Numbers (Linked List)
--------------------------------------------------------------------------------
Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums)):
            check = target - nums[i]
            for j in range(len(nums)):
                if((nums[j] == check) and i!=j):
                    return [i,j]
        return None

--------------------------------------------------------------------------------
String to Integer (atoi)
--------------------------------------------------------------------------------
Input: "4193 with words"
Output: 4193
Explanation: Conversion stops at digit '3' as the next character is not a numerical digit.

class Solution:
    def myAtoi(self, str):
        """
        Time:  O(n)
        Space: O(n)
        """
        if not str: return 0

        str = str.lstrip()
        if not str: return 0

        i = 0
        negative = False
        if str[0] == '+':
            i += 1
        elif str[0] == '-':
            negative = True
            i += 1
        elif not str[0].isdigit():
            return 0

        value = 0
        while i < len(str) and str[i].isdigit():
            value *= 10
            value += int(str[i])
            i += 1
        if negative: value = -value

        INT_MAX = 2**31 - 1
        INT_MIN = -2**31
        if value > INT_MAX:
            return INT_MAX
        elif value < INT_MIN:
            return INT_MIN
        else:
            return value

--------------------------------------------------------------------------------
Container With Most Water
--------------------------------------------------------------------------------
Input: [1,8,6,2,5,4,8,3,7]
Output: 49

class Solution:
    def maxArea(self, height: List[int]) -> int:
        """
        :type height: List[int]
        :rtype: int
        """
        MAX = 0
        x = len(height) - 1
        y = 0
        while x != y:
            if height[x] > height[y]:
                area = height[y] * (x - y)
                y += 1
            else:
                area = height[x] * (x - y)
                x -= 1
            MAX = max(MAX, area)
        return MAX

--------------------------------------------------------------------------------
Most Common Word
--------------------------------------------------------------------------------
Input:
paragraph = "Bob hit a ball, the hit BALL flew far after it was hit."
banned = ["hit"]
Output: "ball"
Explanation:
"hit" occurs 3 times, but it is a banned word.
"ball" occurs twice (and no other word does), so it is the most frequent non-banned word in the paragraph.
Note that words in the paragraph are not case sensitive,
that punctuation is ignored (even if adjacent to words, such as "ball,"),
and that "hit" isn't the answer even though it occurs more because it is banned.

class Solution:
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        banset = set(banned)
        for c in "!?',;.":
            paragraph = paragraph.replace(c, " ")
        count = collections.Counter(
            word for word in paragraph.lower().split())

        ans, best = '', 0
        for word in count:
            if count[word] > best and word not in banset:
                ans, best = word, count[word]

        return ans

--------------------------------------------------------------------------------
 Reorder Log Files
--------------------------------------------------------------------------------
Input: logs = ["dig1 8 1 5 1","let1 art can","dig2 3 6","let2 own kit dig","let3 art zero"]
Output: ["let1 art can","let3 art zero","let2 own kit dig","dig1 8 1 5 1","dig2 3 6"]

class Solution:
    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        def f(log):
            id_, rest = log.split(" ", 1)
            return (0, rest, id_) if rest[0].isalpha() else (1,)

        return sorted(logs, key = f)

--------------------------------------------------------------------------------
Trapping Rain Water
--------------------------------------------------------------------------------
Input: [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6

class Solution:
    def trap(self, height: List[int]) -> int:
        bars = height
        if not bars or len(bars) < 3:
            return 0
        volume = 0
        left, right = 0, len(bars) - 1
        l_max, r_max = bars[left], bars[right]
        while left < right:
            l_max, r_max = max(bars[left], l_max), max(bars[right], r_max)
            if l_max <= r_max:
                volume += l_max - bars[left]
                left += 1
            else:
                volume += r_max - bars[right]
                right -= 1
        return volume

--------------------------------------------------------------------------------
Rotate Image
--------------------------------------------------------------------------------
Given input matrix =
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],

rotate the input matrix in-place such that it becomes:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]
#Transpose the matrix and reverse the rows.
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        #Transpose the matrix and reverse the rows.
        n = len(matrix)
        for i in range(n):
            for j in range(n):
                if i < j:
                    matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        for l in matrix:
            l.reverse()

--------------------------------------------------------------------------------
3Sum
--------------------------------------------------------------------------------

Given array nums = [-1, 0, 1, 2, -1, -4],

A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        N, result = len(nums), []
        for i in range(N):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            target = nums[i]*-1
            s,e = i+1, N-1
            while s<e:
                if nums[s]+nums[e] == target:
                    result.append([nums[i], nums[s], nums[e]])
                    s = s+1
                    while s<e and nums[s] == nums[s-1]:
                        s = s+1
                elif nums[s] + nums[e] < target:
                    s = s+1
                else:
                    e = e-1
        return result

--------------------------------------------------------------------------------
Group Anagrams
--------------------------------------------------------------------------------

Input: ["eat", "tea", "tan", "ate", "nat", "bat"],
Output:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        ans = collections.defaultdict(list)
        for s in strs:
            ans[tuple(sorted(s))].append(s)
        return ans.values()
# Time Complexity: O(NK \log K)O(NKlogK), where NN is the length of strs, and KK is the maximum length of a string in strs. The outer loop has complexity O(N)O(N) as we iterate through each string. Then, we sort each string in O(K \log K)O(KlogK) time.

# Space Complexity: O(NK)O(NK), the total information content stored in ans.


Ans2:
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        hashmap = {}
        for st in strs:
            key = ''.join(sorted(st))
            if key not in hashmap:
                hashmap[key] = [st]
            else:
                hashmap[key] += [st]
        return hashmap.values()

--------------------------------------------------------------------------------
Compare Version Numbers
--------------------------------------------------------------------------------
Input: version1 = "7.5.2.4", version2 = "7.5.3"
Output: -1

Input: version1 = "1.01", version2 = "1.001"
Output: 0
Explanation: Ignoring leading zeroes, both “01” and “001" represent the same number “1”

class Solution:
    def compareVersion(self, version1: str, version2: str) -> int:
        versions1 = [int(v) for v in version1.split(".")]
        versions2 = [int(v) for v in version2.split(".")]
        for i in range(max(len(versions1),len(versions2))):
            v1 = versions1[i] if i < len(versions1) else 0
            v2 = versions2[i] if i < len(versions2) else 0
            if v1 > v2:
                return 1
            elif v1 < v2:
                return -1
        return 0

--------------------------------------------------------------------------------
Rotate Image
--------------------------------------------------------------------------------
Given input matrix =
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],

rotate the input matrix in-place such that it becomes:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]

class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        #Transpose the matrix and reverse the rows.
        n = len(matrix)
        for i in range(n):
            for j in range(n):
                if i < j:
                    matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        for l in matrix:
            l.reverse()

--------------------------------------------------------------------------------
Valid Parentheses
--------------------------------------------------------------------------------
Input: "()[]{}"
Output: true

class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        mapping = {")": "(", "}": "{", "]": "["}
        for char in s:
            if char in mapping:
                # Otherwise assign a dummy value of '#' to the top_element variable
                top_element = stack.pop() if stack else '#'
                if mapping[char] != top_element:
                    return False
            else:
                stack.append(char)
        return not stack

--------------------------------------------------------------------------------
First Unique Character in a String
--------------------------------------------------------------------------------
s = "leetcode"
return 0.

s = "loveleetcode",
return 2.

class Solution:
    def firstUniqChar(self, s: str) -> int:
        # build hash map : character and how often it appears
        count = collections.Counter(s)

        # find the index
        for idx, ch in enumerate(s):
            if count[ch] == 1:
                return idx
        return -1
# Time complexity : \mathcal{O}(N)O(N) since we go through the string of length N two times.
# Space complexity : \mathcal{O}(N)O(N) since we have to keep a hash map with N elements.
--------------------------------------------------------------------------------
Missing Number
--------------------------------------------------------------------------------
Input: [3,0,1]
Output: 2

Input: [9,6,4,2,3,5,7,0,1]
Output: 8

class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        nums.sort()
        if nums[0] == 0:
            if(len(nums) == 1 and nums[0] != 0):
                return nums[0] - 1
            elif(len(nums) == 1 and nums[0] == 0):
                return 1
            else:
                for i in range(len(nums)-1):
                    if(nums[i] + 1 != nums[i+1]):
                        return nums[i] + 1
                return nums[i] + 2
        else:
            return 0

--------------------------------------------------------------------------------
Implement strStr()
--------------------------------------------------------------------------------
Input: haystack = "hello", needle = "ll"
Output: 2

Input: haystack = "aaaaa", needle = "bba"
Output: -1

class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        hay_size = len(haystack)
        needle_size = len(needle)
        pointer = 0
        if(needle_size == 0):
            return 0
        else:
            for i in range(hay_size):
                j = 1
                if(needle[0] == haystack[i]):
                    pointer = i
                    if(needle == haystack[i:i+needle_size]):
                        return pointer
        return -1

--------------------------------------------------------------------------------
3Sum Closest
--------------------------------------------------------------------------------
Given array nums = [-1, 2, 1, -4], and target = 1.
The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
https://leetcode.com/explore/interview/card/amazon/76/array-and-strings/2967/discuss/7913/Python-solution-with-detailed-explanation

Two pointer solution with O(n2) complexity

class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        closest_sum = 2**31-1
        for i in range(len(nums)):
            j,k = i+1, len(nums)-1
            while j<k:
                curr_sum = nums[i] + nums[j] + nums[k]
                if curr_sum == target:
                    return curr_sum
                if abs(curr_sum-target) < abs(closest_sum-target):
                    closest_sum = curr_sum
                if curr_sum < target:
                    j = j+1
                else:
                    k = k-1
        return closest_sum

--------------------------------------------------------------------------------
Product of Array Except Self
--------------------------------------------------------------------------------
Input:  [1,2,3,4]
Output: [24,12,8,6]

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # The length of the input array
        length = len(nums)

        # The answer array to be returned
        answer = [0]*length

        # answer[i] contains the product of all the elements to the left
        # Note: for the element at index '0', there are no elements to the left,
        # so the answer[0] would be 1
        answer[0] = 1
        for i in range(1, length):

            # answer[i - 1] already contains the product of elements to the left of 'i - 1'
            # Simply multiplying it with nums[i - 1] would give the product of all
            # elements to the left of index 'i'
            answer[i] = nums[i - 1] * answer[i - 1]

        # R contains the product of all the elements to the right
        # Note: for the element at index 'length - 1', there are no elements to the right,
        # so the R would be 1
        R = 1;
        for i in reversed(range(length)):

            # For the index 'i', R would contain the
            # product of all elements to the right. We update R accordingly
            answer[i] = answer[i] * R
            R *= nums[i]

        return answer

Time complexity : O(N)O(N) where NN represents the number of elements in the input array. We use one iteration to construct the array LL, one to update the array answeranswer.
Space complexity : O(1)O(1) since don't use any additional array for our computations. The problem statement mentions that using the answeranswer array doesn't add to the space complexity.

--------------------------------------------------------------------------------
 Roman to Integer
--------------------------------------------------------------------------------
Input: "LVIII"
Output: 58
Explanation: L = 50, V= 5, III = 3.

Input: "MCMXCIV"
Output: 1994
Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.

values = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}

class Solution:
    def romanToInt(self, s: str) -> int:
        total = 0
        i = 0
        while i < len(s):
            # If this is the subtractive case.
            if i + 1 < len(s) and values[s[i]] < values[s[i + 1]]:
                total += values[s[i + 1]] - values[s[i]]
                i += 2
            # Else this is NOT the subtractive case.
            else:
                total += values[s[i]]
                i += 1
        return total

Time complexity : O(1)O(1).

As there is a finite set of roman numerals, the maximum number possible number can be 3999, which in roman numerals is MMMCMXCIX. As such the time complexity is O(1)O(1).

If roman numerals had an arbitrary number of symbols, then the time complexity would be proportional to the length of the input, i.e. O(n)O(n). This is assuming that looking up the value of each symbol is O(1)O(1).

Space complexity : O(1)O(1).

Because only a constant number of single-value variables are used, the space complexity is O(1)O(1).

--------------------------------------------------------------------------------
Minimum Window Substring
--------------------------------------------------------------------------------
Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).

Example:

Input: S = "ADOBECODEBANC", T = "ABC"
Output: "BANC"


class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need = collections.Counter(t)            #hash table to store char frequency
        missing = len(t)                         #total number of chars we care
        start, end = 0, 0
        i = 0
        for j, char in enumerate(s, 1):          #index j from 1
            if need[char] > 0:
                missing -= 1
            need[char] -= 1
            if missing == 0:                     #match all chars
                while i < j and need[s[i]] < 0:  #remove chars to find the real start
                    need[s[i]] += 1
                    i += 1
                need[s[i]] += 1                  #make sure the first appearing char satisfies need[char]>0
                missing += 1                     #we missed this first char, so add missing by 1
                if end == 0 or j-i < end-start:  #update window
                    start, end = i, j
                i += 1                           #update i to start+1 for next window
        return s[start:end]

--------------------------------------------------------------------------------
Minimum Window Substring
--------------------------------------------------------------------------------
