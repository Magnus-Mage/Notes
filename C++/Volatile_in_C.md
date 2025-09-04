## Explaning use case of volatile declaration in C

`volatile` tells the compiler not to optimize anything that has to do with the volatile variable.

There are at least three common reasons to use it, all involving situations where the value of the variable can change without action from the visible code:

- When you interface with hardware that changes the value itself
- When there's another thread running that also uses the variable
- When there's a signal handler that might change the value of the variable.

Let's say you have a little piece of hardware that is mapped into RAM somewhere and that has two addresses: a command port and a data port:

```c
typedef struct
{
  int command;
  int data;
  int isBusy;
} MyHardwareGadget;
```

Now you want to send some command:

```c
void SendCommand(MyHardwareGadget* gadget, int command, int data)
{
  // wait while the gadget is busy:
  while (gadget->isBusy)
  {
    // do nothing here.
  }
  // set data first:
  gadget->data    = data;
  // writing the command starts the action:
  gadget->command = command;
}
```

Looks easy, but it can fail because the compiler is free to change the order in which data and commands are written. This would cause our little gadget to issue commands with the previous data-value. Also take a look at the wait while busy loop. That one will be optimized out. The compiler will try to be clever, read the value of isBusy just once and then go into an infinite loop. That's not what you want.

The way to get around this is to declare the pointer gadget as volatile. This way the compiler is forced to do what you wrote. It can't remove the memory assignments, it can't cache variables in registers and it can't change the order of assignments either

This is the correct version:

```c
void SendCommand(volatile MyHardwareGadget* gadget, int command, int data)
{
  // wait while the gadget is busy:
  while (gadget->isBusy)
  {
    // do nothing here.
  }
  // set data first:
  gadget->data    = data;
  // writing the command starts the action:
  gadget->command = command;
}
```
