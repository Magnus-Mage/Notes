# RValues Move Semantics in C++

## The motivation for Move Semantics

Move semantics was added in C++11 to efficiently move dead / RValues instead of copying them, making the whole process much more efficient and senseful.

Take this below example to udnerstand why move semantics is way better than copy semantics in a lot of cases.

```c++
// Example 1
Godzilla g1 = factory.createFrieghteningGodzilla();

// Example 2
Godzilla g2;
g2 = factor.createSpookyGodzilla();

// Example 3
list<Godzilla> godzillas;
godzillas.push_back(Godzilla("sweety"));
```

Let's assume you have a factory that can create a godzilla, we can assume from here, we can assume that the factory is creating a local godzilla internally and then returning them by value.

And then, when we get the godzilla created by the constructor g1, we use copy constructor, which is not efficient (assuming Godzilla is a big object).

So, instead we can also take/steal the assets of this object instead of copying it.

> Avoid redundant copying when we can move from an almost dead object.
> * Some of the redundant inefficient copies can be avoided by the compiler using RVO(Return Value Optimisation) or NRVO(Named Return Value Optimisation). But not at all.

### Question (1)

Q. Should we move here?
```c++
std::string str2 = str1;
```

The answer would be no, as str1 can still be called somewhere else after this and therefore, not a dead object.

### Question (2)
 Q. Should we move this?
```c++
std::string str3 = str2 + str1;
```

The answer would be yes, as `str1 + str2` is an dead object. Honestly, calling it a dead object would be wrong, it's more related to the lifetime of the object.
A compiler can see the lifetime of the object and if the object ends after the current expression, then its better to use move semantics.

So, how can we distinguish between the cases?

```c++
std::string str2 = str1;
std::string str3 = str2 + str1;
```

- Both assigned expresiion are refrences
- Currently, both lines would call the copy constructor.
- But they are kind of different refrences:
	- The reference at first **would still be alive after the end of the statement**
	- The refrence at second **would not, it is a temporary object.**
