# Scope :
- When you declare something in C++(like variables, functions, or class), that declaration is visible(accesible) only in certain parts of the code.
- A scope is a region of the program where names can be accessed, like inside a function, inside a class or at a global level.
- **A discontiguous scope** meaning the parts of the program where a declaration is visible don't have to be a continuous block. The declaration might be visible in some other place, but not others, even if places are seperated.

Example:
```c++
  int x;  // visible in the global scope

void f() {
    // x is visible here
}

void g() {
    // x is also visible here
}

// But inside a different namespace or class, x might not be visible.
  ```
- Within a scope, unqualified name lookup can be used to associate a name with its declaration.
```c++
int x = 5; // global x

void f() {
    int x = 10; // local x shadows global x
    // Inside this scope, unqualified lookup for 'x' finds the local one (10)
}
```
