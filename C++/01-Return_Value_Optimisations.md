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
Now coming back to the Main topic of RVO, we can see why RVO matters and why [Move semantics](https://stackoverflow.com/questions/3106110/what-is-move-semantics) might not be the best replacement.

So, in our previous [talk](00-Return_Value_Optimisations.md), we used move semantics in example two to show how it works. 
```c++
// Move constructor
Foo::Foo(Foo&& rhs)
{
    std::cout << "Move ctor\n";
}
```
By adding a move constructor, our output without eliding constructors will now be as follows.

```bash
$ ./appnoelide
Default ctor
Move ctor
Move ctor
Default ctor
Move ctor
Move ctor
```
Move semantics is really helpful as it's less costly than a copy.
But, it also has it's own limitations.

Take this example here ->
<img width="408" height="486" alt="image" src="https://github.com/user-attachments/assets/a5de54b8-ed47-4eec-8188-23a8ed32bd00" />

Here, we are using move semantics for a loop with 10000 cycles that can lead to 30000 moves being called. (generally depends on compiler and version of the compiler)

So, in an overview for a large codebase, it will start hampering the performance of the code. 

> Results for the same.
<img width="1062" height="595" alt="image" src="https://github.com/user-attachments/assets/140e3baf-dac3-4de2-a655-437643241333" />


