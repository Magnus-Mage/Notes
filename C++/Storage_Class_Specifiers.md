# Agenda
- Storage class specifiers defination and useage.
- What is **Storage Usage**?
- Types of specifiers
- Linkage

## Storage Class Specifiers:
The storage class specifiers are a part of the `decl-specifier-seq` of a name's declaraction syntax.
Together with the [scope](https://en.cppreference.com/w/cpp/language/scope.html) of the name, they control the independent properties of the name: 
- Storage Duration and
- Linkage

### Storage duration
The storage duration is the property of an object that defines the minimum lifetime potential of the storage containing the object. The storage duration is determined by the construct used to create the object and is one of the following:
- static storage duration
- thread storage duration (related to thread and thread lifetime)
- automatic storage duration (exists for its own block scope)
- dynamic storage duration (created using [new expressions](https://en.cppreference.com/w/cpp/language/new.html))

> Thread and automatic storage durations are associated with objects introduced by declarations and with temporary objects.
> 
> The dynamic storage duration is associated with objects created by new expression or with implicitly created objects.

### Specifiers
The following keywords are storage class specifiers
- auto
- register (deprecated after C++17; used for keeping the object int he register for faster retrival)
- static (persists throughout the program)
- thread_local (thread lifetime)
- extern (including external variables and persists throughout the program lifetime)
- mutable (has no actual affect on storage duration but still a part of the same; eg: const/ [volatile](Volatile_in_C.md)

> register is a hint that the variable so declared will be heavily used, so that its value can be stored in a CPU register. The hint can be ignored, and in most implementations it will be ignored if the address of the variable is taken. This use is deprecated.

**Thread Storage duration** : All variables declared with thread_local have thread storage duration. The storage of this entities lasts for the duration of the thread in which they were created.

**Static storage duration** : 
A variable satisfying all following conditions has static storage duration ﻿:

- It belongs to a namespace scope or are first declared with static or extern.
- It does not have thread storage duration. (this is simply cause they persists during the entire program and thus, the involve the entire duration of the program and no comparisions can be made)
  
The storage for these entities lasts for the duration of the program.

**Automatic storage duration** :
The following variables have automatic storage duration ﻿:

- Variables that belong to a block scope and are not explicitly declared static, thread_local,(since C++11) or extern. The storage for such variables lasts until the block in which they are created exits.
- Variables that belong to a parameter scope (i.e. function parameters). The storage for a function parameter lasts until immediately after its destruction.
**Dynamic storage duration** :
Objects created by the following methods during program execution have dynamic storage duration ﻿:

- new expressions. The storage for such objects is allocated by allocation functions and deallocated by deallocation functions.
- Implicitly creation by other means. The storage for such objects overlaps with some existing storage.
- Exception objects. The storage for such objects is allocated and deallocated in an unspecified way.

## Linkage:
A name can have external linkage, modile linkage, internal linkage, or no linkage:
- A entity whose name has enternal linkage can be redeclared in another C++ file (translation unit), and the redeclaration can be attached to a different module.
- An entity whose name has module linkage can be redeclared in another C++ file, as long as the redeclaration is attached to the same module.
- An entity whose name has internal linkage can be redeclared in another scope in same C++ file.
- An entity whose name has no linkage can be redeclared in the same scope.

**No linkage** :
Any of the following names declared at block scope have no linkage:

- variables that are not explicitly declared extern (regardless of the static modifier);
- local classes and their member functions;
- other names declared at block scope such as typedefs, enumerations, and enumerators.
Names not specified with external, module,(since C++20) or internal linkage also have no linkage, regardless of which scope they are declared in.

Example:
```c++
// file.cpp
void func() {
    int a = 1; // no linkage

    {
        int a = 2; // okay: different a, no linkage
    }
}
```

**Internal linkage** :
Any of the following names declared at namespace scope have internal linkage:

- variables, variable templates(since C++14), functions, or function templates declared static;
- non-template (since C++14)variables of non-volatile const-qualified type, unless
    - they are inline,
    - they are declared in the purview of a module interface unit (outside the private module fragment, if any) or module partition,
    - they are explicitly declared extern, or
    - they were previously declared and the prior declaration did not have internal linkage;
- data members of anonymous unions.
In addition, all names declared in unnamed namespaces or a namespace within an unnamed namespace, even ones explicitly declared extern, have internal linkage.

Example:
```c++
// file.cpp
static int z = 5; // internal linkage

void func() {
    static int z = 10; // a different z: same name, but allowed (same file)
}

```

**External linkage** :
Variables and functions with external linkage also have language linkage, which makes it possible to link translation units written in different programming languages.

Any of the following names declared at namespace scope have external linkage, unless they are declared in an unnamed namespace or their declarations are attached to a named module and are not exported(since C++20):

- variables and functions not listed above (that is, functions not declared static, non-const variables not declared static, and any variables declared extern);
- enumerations;
- names of classes, their member functions, static data members (const or not), nested classes and enumerations, and functions first introduced with friend declarations inside class bodies;
- names of all templates not listed above (that is, not function templates declared static).
Any of the following names first declared at block scope have external linkage:

- names of variables declared extern;
- names of functions.

Example:
```c++
// file1.cpp
int x = 42; // external linkage by default

// file2.cpp
extern int x; // valid: refers to x from file1.cpp
```

**Module linkage** :
Names declared at namespace scope have module linkage if their declarations are attached to a named module and are not exported, and do not have internal linkage.

Example:
```c++
// my_module.ixx
export module my_module;
module :private;

int y = 100; // module linkage (not exported)

// file.cpp
import my_module;
// y is **not** visible here because it has module linkage
```
