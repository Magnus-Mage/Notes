# Return Value Optimisations for performance in C++ Codebases

## Agenda
- What is RVO?
- Anti-patterns that prevent RVO (some examples)
- Use RVO for performance gains

## Some history

RVO first made its way to windows compiler in [Zortech](https://en.wikipedia.org/wiki/Digital_Mars) in 1988.

Later on, a proposal was made in 1997 C++ meeting to clarify [Copy Elision](https://en.cppreference.com/w/cpp/language/copy_elision.html) and include it in the standard. 

## What is Return value optimisations?
> RVO is actually not defined in the standard but specified under **Copy Elision**.

Return Value optimisation(RVO) in C++ is a compiler optimsiation that eliminates the creation of temporary objects by directly contructing the return value in the memory location of the caller.

> Example
<img width="329" height="250" alt="image" src="https://github.com/user-attachments/assets/2fb7f552-3c5e-4d35-a552-e8b3edbabf55" />

Here, on `line 2` we are initialising our object `a` and then returning it on `line 3`. Now, we have a local variable on `line 8` -> `e1` -> which is being initialised with `example1()` will first make the object at `a`
and then **copy/move** that value to the object and finally to the local variable `e1`. Thats the case for a non-RVO optimisated code, with RVO, we can move the value/object directly to the variable without any extra steps.
Therefore, we are using **Copy/Move elision** and thus the name.

> Assembly perspective for x86_64 comparision.
<img width="1061" height="597" alt="image" src="https://github.com/user-attachments/assets/4f590265-c673-471f-b5e7-b479d13b8800" />


> Another example to make the concept clear.
```c++
#include <iostream>

class Foo
{
public:
    int x = 0;

    // default ctor
    Foo()
    {
        std::cout << "Default ctor\n";
    }

    // copy ctor
    Foo(const Foo& rhs)
    {
        std::cout << "Copy ctor\n";
    }
};

Foo CreateFooA()
{
    return Foo();
}

Foo CreateFooB()
{
    Foo temp;
    temp.x = 42; // update member variable
    return temp;
}

int main()
{
    Foo t1(CreateFooA());
    Foo t2(CreateFooB());
   
    return 0;
}
```

If we compile the code above and run the executable, we get the following output (note that we are compiling in C++14, more on this later).
```bash
$ g++ -std=gnu++14 copyelision.cpp -o app
$ ./app
Default ctor
Default ctor
```


Now, if we compile the same code, but this time we explicitly telling our compiler to not elide constructors using `-fno-elide-constructors`, we will get the following output.
```bash
$ g++ -std=gnu++14 -fno-elide-constructors copyelision.cpp -o appnoelide
$ ./appnoelide
Default ctor
Copy ctor
Copy ctor
Default ctor
Copy ctor
Copy ctor
```
Woah! That is a lot more output than when we compiled it with default settings! Let’s take a look at why this is the case.

```c++
Foo t1(CreateFooA());
```

When we declare t1 and initialised it with the returned rvalue from CreateFooA(). The following will happen:

CreateFooA() creates a temporary Foo object to return.
The temporary object will then be copied into the object that will be returned by CreateFooA().
The value returned by CreateFooA() will then be copied into t1.
This is tedious and would adversely affect performance if the object we are dealing with is expensive to copy!

```c++
Foo t2(CreateFooB());
```

The initialisation of t2 goes through a similar process:

CreateFooB() creates a temporary Foo variable, temp.
Do things with temp. In our case, we update the member variable temp.x to 42.
temp then has to be copied into the value to be returned by CreateFooB().
The value returned by CreateFooB() will then be copied into t2.
Again, very tedious and inefficient. In short, the whole process can be summarised as the following.

```c++
$ ./appnoelide
Default ctor # construct temporary variable to return in CreateFooA()
Copy ctor    # copying temporary variable to returning object of CreateFooA()
Copy ctor    # copying rvalue returned by CreateFooA() into t1
Default ctor # construct 'temp' inside CreateFooB()
Copy ctor    # copy 'temp' into to returning object of CreateFooB()
Copy ctor    # copying rvalue returned by CreateFooB() into t2
```

C++11’s move semantics help improve this by avoiding copying resources, but moving them instead.

```c++
// Move constructor
Foo::Foo(Foo&& rhs)
{
    std::cout << "Move ctor\n";
}
```
By adding a move constructor, our output without eliding constructors will now be as follows.

```c++
$ ./appnoelide
Default ctor
Move ctor
Move ctor
Default ctor
Move ctor
Move ctor
```
But wait, CreateFooB() returns a named variable, which means it is an lvalue. Why is it able to call on the move constructor?

This is because of a rule in the C++ standard, specifically in 12.8. It states that when a function has a class return type and criteria for copy elision is met, if the expression to be returned is a named variable (lvalue), the object is treated as an rvalue when selecting the constructor for the copy i.e. move constructor will be used if it is available.

This makes uncopyable objects like std::unique_ptr able to be returned by value even if it is a named variable.

```c++
std::unique_ptr<int> CreateUnique()
{
    auto ptr = std::make_unique<int>(0);
    return ptr; // This compiles!
}
```
It is important to note though, that if a copy or move constructor is elided, that constructor must still exist — This ensures that the optimisation still respects whether an object is of a class that is uncopyable or unmovable.

Now, if we go back to the output of the executable with copy elision optimisation enabled — remember that it is enabled by default!

```c++
$ ./app
Default ctor
Default ctor
```
Here we elide extra calls to constructors, and directly construct the “to be returned” objects to the variable it is assigned to.
