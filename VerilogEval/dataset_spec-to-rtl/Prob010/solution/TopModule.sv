module TopModule (
    input  x,
    input  y,
    output z
);

    wire xor_out;

    XOR_Module u_XOR (
        .x(x),
        .y(y),
        .xor_out(xor_out)
    );

    AND_Module u_AND (
        .a(xor_out),
        .b(x),
        .z(z)
    );

endmodule