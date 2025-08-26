# Types of RVO

## Named Return Value Optimisation (NRVO) and Unnamed Return Value Optimisation

<img width="1056" height="581" alt="image" src="https://github.com/user-attachments/assets/f61d5951-8bd0-45e9-8149-babf1daa8f6f" />

In example 1,
            we are returning [lvalue](https://stackoverflow.com/questions/3601602/what-are-rvalues-lvalues-xvalues-glvalues-and-prvalues) which is a named variable -> thus, Named Return Value Optimisation.

In example 2,
            we are returning a [prvalue](https://stackoverflow.com/questions/3601602/what-are-rvalues-lvalues-xvalues-glvalues-and-prvalues) , which is a temporary instance -> thus, Unnamed Return Value Optimisation.


## Does my compiler have to support RVO?
- Compilers are allowed to perform URVO since C++98.
- Compilers are required to provide URVO support since C++17.
- NVRO support is optional, but recommended.

We can actually turn off (N)RVO for our compiler using -> `-fno-elide-constructors`.

> MSVC actually has NVRO by default and needs to be enabled instead, using `-O2`.

# RVO use case:
Now coming back to the Main topic of RVO, we can see why RVO matters and why move semantics might not be the best replacement.

So, in our previous [talk](00-Return_Value_Optimisations.md], we used move semantics in example two to show how it works. 
