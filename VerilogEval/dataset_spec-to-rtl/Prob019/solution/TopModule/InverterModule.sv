module InverterModule (
    input wire a,
    output wire y
);
    assign y = ~a;
endmodule