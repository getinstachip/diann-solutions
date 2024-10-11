module OutputLogic (
    input  logic [2:0] state,
    output logic       disc,
    output logic       flag,
    output logic       err
);

    always_comb begin
        // Default output values
        disc = 1'b0;
        flag = 1'b0;
        err  = 1'b0;
        
        case (state)
            3'b001: disc = 1'b1; // State representing 0111110
            3'b010: flag = 1'b1; // State representing 01111110
            3'b100: err  = 1'b1; // State representing 01111111...
            default: begin
                disc = 1'b0;
                flag = 1'b0;
                err  = 1'b0;
            end
        endcase
    end

endmodule