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

## General:
Each program in C++ has a **global** scope, which contains the entire program.
Each other scope S is introduced by the following:
- declaration
- a parameter in parameter list
- a statement
- a handler
- a [contact assertion](https://en.cppreference.com/w/cpp/language/contracts.html) (C++26 feature)

S always appear in other scopes, which thereby contains S by default.

An **enclosing scope** at a program is any scope that contains it; the smallest such scope is said to be the **immediate scope** at that point.

example:
```c++
int x = 10;    // Global scope S

void foo()
{
    int x = 20;  // local scope; otherwise the immediate scope for variable x
}
```

- If you have a box S somewhere, and a point P outside S, then any box that contains S but not P is an "intervening" scope between P and S.
- Meaning, if `P` is a program point located in `f()` after the inner block, the inner block scope S itself intervenes between P and S (trivially)

An entity belongs to a scope S if S is the target scope of a declaration of the entity.

```c++
//                global  scope  scope
//                scope     S      T
int x;         //   ─┐                 // program point X
               //    │
{              //    │     ─┐
    {          //    │      │     ─┐
        int y; //    │      │      │   // program point Y
    }          //    │      │     ─┘
}              //   ─┘     ─┘
```

In the program above:

- The global scope, scope S and scope T contains program point Y.
  - In other words, these three scopes are all enclosing scopes at program point Y.
- The global scope contains scopes S and T, and scope S contains scope T.
  - Therefore, scope T is the smallest scope among all three, which means:
    - Scope T is the immediate scope at program point Y.
    - The declaration of the variable y inhabits scope T at its locus.
    - Scope T is the target scope of the declaration of y.
    - The variable y belongs to scope T.
  - Scope S is the parent scope of scope T, and the global scope is the parent scope of scope S.
- Scope S intervenes between program point X and scope T.

### Block Scope:

Each 
- Selection statement (if, switch)
- iteration statement (for, range-for, while, do-while)
- handler, or
- compound statement (nested loops, mix of all the above ones)

introduces a *block scope* that includes the statement or handler. A variable that belongs to a block scope is a `block variable`(also known as local variable).
```c++
int i = 42;
int a[10];
 
for (int i = 0; i < 10; i++) // inner “i” inhabits the block scope
    a[i] = i;                // introduced by the for-statement
 
int j = i; // j = 42
```

Do Note, that even if we declare some variable or function or uses the extern specifier in a block scope S, the declaration shall not be attached to a named module, its target scope is a larger enclosing scope, but the name is bound in their mediate scope S.
```c++
if (int x = f())  // declares “x”
{ // the if-block is a substatement of the if-statement
    int x;        // error: redeclaration of “x”
}
else
{ // the else-block is also a substatement of the if-statement
    int x;        // error: redeclaration of “x”
}
 
void g(int i)
{
    extern int i; // error: redeclaration of “i”
}
```

## Function parameter scope
Each parameter declaration P introduces a function parameter scope that includes P.

- If the declared parameter is of the parameter list of a function declaration:
  - If the function declaration is a function definition, the scope introduced is extended to the end of the function definition.
  - Otherwise (the function declaration is a function prototype), the scope introduced is extended to the end of the function declarator.
  - In both cases, the scope does not include the locus of the function declaration.
- If the declared parameter is of the parameter list of a lambda expression, the scope introduced is extended to the end of *{ body }*.
- If the declared parameter is of the parameter list of a deduction guide, the scope introduced is extended to the end of that deduction guide.
- If the declared parameter is of the parameter list of a requires expression, the scope introduced is extended to the end of *{ requirement-seq }*.

```c++
int f(int n) // the declaration of the parameter “n”
{            // introduces a function parameter scope
    /* ... */
}            // the function parameter scope ends here
```
