module TopModule (
    input  [2:0] in,
    output [1:0] out
);
    wire sum1, carry1;
    wire sum2, carry2;

    // Instantiate first Adder
    Adder adder1 (
        .a(in[0]),
        .b(in[1]),
        .sum(sum1),
        .carry(carry1)
    );

    // Instantiate second Adder
    Adder adder2 (
        .a(sum1),
        .b(in[2]),
        .sum(sum2),
        .carry(carry2)
    );

    // Glue code to compute the final output
    assign out = {carry1 | carry2, sum2};
endmodule