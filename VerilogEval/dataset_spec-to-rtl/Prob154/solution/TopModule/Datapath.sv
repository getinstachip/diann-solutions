module Datapath (
    input wire clk,
    input wire reset,
    input wire [7:0] in,
    output reg [23:0] out_bytes
);

    always @(posedge clk) begin
        if (reset) begin
            out_bytes <= 24'd0;
        end else begin
            out_bytes <= {out_bytes[15:0], in};
        end
    end

endmodule