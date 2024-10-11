module Mux256to1 (
    input  wire [255:0] in,
    input  wire [7:0]   sel,
    output wire         out
);
    assign out = in[sel];
endmodule