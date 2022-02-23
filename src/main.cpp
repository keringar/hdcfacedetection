#include <fmt/format.h>

int main(int argc, char* argv[])
{
    if (argc) {
        fmt::print("Starting {}\n", argv[0]);
    }

    return 0;
}

