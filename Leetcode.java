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
