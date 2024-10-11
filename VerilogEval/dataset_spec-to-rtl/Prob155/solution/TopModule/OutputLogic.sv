module OutputLogic (
    input  logic [2:0] state,
    input  logic       fall_done,
    output logic       walk_left,
    output logic       walk_right,
    output logic       aaah,
    output logic       digging
);

    always @(*) begin
        // Default assignments
        walk_left  = 1'b0;
        walk_right = 1'b0;
        aaah       = 1'b0;
        digging    = 1'b0;
        
        case (state)
            3'b000: begin // WALK_LEFT
                walk_left  = 1'b1;
            end
            3'b001: begin // WALK_RIGHT
                walk_right = 1'b1;
            end
            3'b010: begin // FALLING
                aaah = 1'b1;
            end
            3'b011: begin // DIGGING
                digging = 1'b1;
            end
            3'b100: begin // SPLATTER
                // All outputs remain 0
            end
            default: begin
                // Ensure all outputs are 0 for undefined states
            end
        endcase
    end

endmodule