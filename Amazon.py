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
