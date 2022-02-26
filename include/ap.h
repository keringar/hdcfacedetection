#pragma once

#include <cstdint>
#include <cassert>

#define BIT_WIDTH 2
#define CONTAINER_TYPE int8_t

// Saturating Arbitrary Precision Integer
class AP {
  public:
    AP() : data(0) 
    { }

    // Normal Constructor
    AP(CONTAINER_TYPE val) : data(val) {
      assert(val >= -(1 << (BIT_WIDTH-1)) + 1);
      assert(val <= (1 << (BIT_WIDTH-1)));
    }

    // Copy Constructor
    AP(const AP& obj) : data(obj.data)
    { }

    // Move Constructor
    AP(AP&& o) noexcept : data(std::move(o.data))
    { }

    CONTAINER_TYPE val() {
      return data;
    }

    // prefix increment
    AP& operator++()
    {
        if (data != (1 << (BIT_WIDTH-1))) {
          data++;
        }

        return *this; // return new value by reference
    }
 
    // postfix increment
    AP operator++(int)
    {
        AP old = *this; // copy old value
        operator++();  // prefix increment
        return old;    // return old value
    }
 
    // prefix decrement
    AP& operator--()
    {
        if (data != -(1 << (BIT_WIDTH-1)) + 1) {
          data--;
        }

        return *this; // return new value by reference
    }
 
    // postfix decrement
    AP operator--(int)
    {
        AP old = *this; // copy old value
        operator--();  // prefix decrement
        return old;    // return old value
    }

  private:
    CONTAINER_TYPE data;
};