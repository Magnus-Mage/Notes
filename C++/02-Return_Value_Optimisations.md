# When will RVO fail to implement implicitly by compiler

- First, would be the flag `-fno-elide-contructors` for C++14 and below which will disregard the optimisation. (For C++17 and above , only NVRO is ignored)
  
- When the object construction happens outside the scope of the current fucntion.
<img width="423" height="288" alt="image" src="https://github.com/user-attachments/assets/19af6110-5402-40d3-96b8-c77c153eed95" />

- When the return type is not the same as what's being returned
<img width="464" height="363" alt="image" src="https://github.com/user-attachments/assets/473f58f5-2cce-4810-aee8-c6690697b57f" />

- When there are multiple return statements returning different objects (For NVRO only) -> the compiler doesn't know which branch to optimise and thus, ignores it.
 <img width="445" height="382" alt="image" src="https://github.com/user-attachments/assets/c25e3a49-9791-432b-ad72-2f4e6dcdd6af" />

- When you are using a complex expression (For NVRO only) -> Not a lvalue or prvalue.
<img width="453" height="181" alt="image" src="https://github.com/user-attachments/assets/093bc190-723e-47b2-92d3-0ec6e40bb0ec" />

- A RVO optimisation wont work for a class where copy and move constructors do not exist, thus, compiler will throw error. But, if the object is directly returned then it will compile and use URVO optimisation instead for a prvalue.


# Conclusion

Actually that's about it, you can check out the [original video](https://www.youtube.com/watch?v=WyxUilrR6fU) after this for different examples on the topic. 
