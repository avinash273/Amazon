/*----------------------------------------------------------------------------*/
//Reverse a character
//Time complexity O(n)
//Space complexity O(1)
class Solution {
    public void reverseString(char[] s) {
        int left = 0, right = s.length-1;
        char temp;
        while(left < right){
            temp =  s[left];
            s[left] = s[right];
            s[right] = temp;
            left++;
            right--;
        }
    }
}
/*----------------------------------------------------------------------------*/
//Two sum
//Time complexity O(n)
//Space complexity O(n)
/*
Given nums = [2, 7, 11, 15], target = 9
Because nums[0] + nums[1] = 2 + 7 = 9
return [0, 1].
*/
class Solution {
    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> checkSum = new HashMap<>();

        for (int i=0; i<nums.length; i++){
            int balance = target - nums[i];
            if(checkSum.containsKey(balance)){
                return new int[] {checkSum.get(balance),i};
            }
            checkSum.put(nums[i], i);
        }
        return null;
    }
}
/*----------------------------------------------------------------------------
Longest Substring
//Time complexity O(n)
//Space complexity O(n)
Input: "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
*/
public int lengthOfLongestSubstring(String s) {
        if (s.length() == 1)
            return 1;
        else if (s.length() == 0)
            return 0;
        char[] tokens = s.toCharArray();
        HashMap<Character, Integer> storeChar = new HashMap<>();
        int counter = 0, i = 0, j = 0, max_val = 0;
        int strLen = tokens.length;
        while (i < strLen) {
            if (storeChar.containsKey(tokens[i])) {
                i = storeChar.get(tokens[i]) + 1;
                storeChar.clear();
                counter = 0;
            } else {
                storeChar.put(tokens[i], i);
                counter++;
                max_val = Math.max(counter, max_val);
                i++;
            }

        }
        System.out.println(storeChar);
        System.out.println(max_val);
        return max_val;
    }
/*----------------------------------------------------------------------------
